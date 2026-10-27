"""GPU pass of the CRPS-paper radio evaluation: draw N posterior samples per test image and stream
per-image statistics to an .npz (bootstrap/aggregation happens in eval_analyse.py on CPU).

Paired design: the same test images (first --n-test of data/test, fixed observation noise stored in the
data), and latent seeds torch.manual_seed(1000 + i) with identical chunking for every model.
All statistics are in sky units (x / max(x) = 1 peak), generator in eval mode (as in riGAN validation).

Usage: python eval_sample.py --ckpt <path> --tag R [--n-test 1000 --n-samp 128]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-radio"
sys.path.insert(0, ROOT)
from models.lightning.riGAN import riGAN  # noqa: E402
from proper_scores import fair_crps, ms_fair_crps, fair_energy_score  # noqa: E402

DATA = "/share/gpu0/jjwhit/crps_review/paper/radio/data/"
OUT = "/share/gpu0/jjwhit/crps_review/paper/radio/eval/"
H = 128
PATCH = [1, 2, 4, 8, 16, 32, 64, 128]
K_EDGES = np.array([1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91], dtype=float)  # 12 radial |k| bins (91 = corner)
VS_LAGS = [(0, 1), (1, 0), (1, 1), (1, -1)]
FIG_IMAGES = [0, 1, 2, 3]
MASK_REL = 0.1  # source mask (y-measurable, same for all models): Gaussian-smoothed (sigma 2 px) dirty > 0.1 * max(dirty), dilated 2 px


def gauss_smooth(img, sigma):
    f = torch.fft.fftfreq(H, device=img.device)
    k = torch.exp(-2 * np.pi ** 2 * (f[:, None] ** 2 + f[None, :] ** 2) * sigma ** 2)
    return torch.fft.ifft2(torch.fft.fft2(img) * k).real


def source_mask(dirty):
    m = (gauss_smooth(dirty, 2.0) > MASK_REL * dirty.max()).float()
    return F.max_pool2d(m[None, None], 5, stride=1, padding=2)[0, 0] > 0


class Spec:
    def __init__(self, dev):
        f = torch.fft.fftfreq(H, d=1.0 / H).to(dev)
        kr = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2).flatten()
        idx = torch.bucketize(kr, torch.tensor(K_EDGES, device=dev, dtype=kr.dtype), right=True) - 1
        nb = len(K_EDGES) - 1
        M = torch.zeros(H * H, nb, device=dev)
        ok = (idx >= 0) & (idx < nb)
        M[ok.nonzero()[:, 0], idx[ok]] = 1.0
        self.M = M / M.sum(0, keepdim=True)
        self.k_eff = (kr[:, None] * self.M).sum(0).cpu().numpy()

    def __call__(self, X):  # (B, H, W) -> (B, nb)
        Fx = torch.fft.fft2(X)
        return ((Fx.real ** 2 + Fx.imag ** 2) / (H * H)).reshape(X.shape[0], -1) @ self.M


def patch_means(X, s):
    n = H // s
    return X.reshape(X.shape[0], n, s, n, s).mean((2, 4)).reshape(X.shape[0], -1)


def ranks(S, t):
    """Rank of truth t (D,) among samples S (N, D): number of samples strictly below, ties randomised."""
    below = (S < t).sum(0)
    eq = (S == t).sum(0)
    if eq.any():
        below = below + (torch.rand_like(eq, dtype=torch.float) * (eq + 1)).floor().long()
    return below


def variogram(S, x, p=0.5):
    N = S.shape[0]
    out = []
    for dy, dx in VS_LAGS:
        ys = (slice(0, H - dy), slice(dy, H))
        xs = (slice(0, H - dx), slice(dx, H)) if dx >= 0 else (slice(-dx, H), slice(0, H + dx))
        a = (S[:, ys[0], xs[0]] - S[:, ys[1], xs[1]]).abs().double() ** p
        t = (x[ys[0], xs[0]] - x[ys[1], xs[1]]).abs().double() ** p
        out.append((((t - a.mean(0)) ** 2) - a.var(0) / N).mean().item())
    return out


def ssim_psnr(pred, gt):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    p, g = pred.cpu().numpy().astype(np.float64), gt.cpu().numpy().astype(np.float64)
    return (peak_signal_noise_ratio(g, p, data_range=g.max()),
            structural_similarity(g, p, data_range=g.max() - g.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-samp", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device

    model = riGAN.load_from_checkpoint(a.ckpt, map_location="cpu").to(dev).eval()
    model.args.num_workers = 0
    print("reg", model.reg, "std_mult", model.std_mult, "epoch ckpt", a.ckpt, flush=True)
    tr = DATA + "train/"
    mx, sx, my, sy, mu, su = [float(np.load(tr + f"{k}.npy")) for k in
                              ["mean_x", "std_x", "mean_y", "std_y", "mean_uv", "std_uv"]]
    X = np.load(DATA + "test/x.npy")[: a.n_test]
    Y = np.load(DATA + "test/y.npy")[: a.n_test]
    UV = np.load(DATA + "test/uv.npy")[: a.n_test]
    spec = Spec(dev)
    N = a.n_samp
    rec = {k: [] for k in ["psnr", "ssim", "rank_pix_all", "rank_pix_src", "rank_pix_bg", "rank_pix_tru", "rank_pix_zero",
                           "rank_clip_all", "rank_clip_src", "rank_clip_bg", "rank_clip_tru", "rank_clip_zero",
                           "mask_frac", "patch_rank", "patch_src_rank", "pow_samp", "pow_true", "pow_mean",
                           "pow_cross", "pow_rank", "pow_logsamp", "es", "vs", "crps", "mscrps", "v", "e", "e_pix_src",
                           "v_pix_src", "pair_d", "truth_d", "near_ident", "pix_pit_absdev", "flux", "time"]}
    figs = {}
    with torch.no_grad():
        for i in range(len(X)):
            t0 = time.time()
            x = torch.from_numpy(X[i]).float().to(dev)
            d = torch.from_numpy(Y[i]).float().to(dev)
            y = torch.stack([(d - my) / sy, (torch.from_numpy(UV[i]).float().to(dev) - mu) / su])[None]
            torch.manual_seed(1000 + i)
            S = torch.cat([model(y.repeat(min(a.chunk, N - j), 1, 1, 1)) for j in range(0, N, a.chunk)])[:, 0]
            S = S * sx + mx  # (N, H, W) sky units
            m = S.mean(0)
            # 1. reconstruction
            ps, ss = zip(*[ssim_psnr(S[:P].mean(0), x) for P in (1, 8, 32)])
            rec["psnr"].append(ps)
            rec["ssim"].append(ss)
            # 2. marginal calibration: per-pixel ranks (histogram over N+1 values)
            msk = source_mask(d)
            tru = x > 0
            r = ranks(S.reshape(N, -1), x.reshape(-1)).reshape(H, H)
            # positivity-clipped samples (monotone projection onto the sky support x >= 0): ties at 0 are randomised,
            # which gives the randomised PIT of the atom at 0 (truth is exactly 0 on ~86% of pixels)
            rc = ranks(S.clamp_min(0).reshape(N, -1), x.reshape(-1)).reshape(H, H)
            for k, sel in [("all", torch.ones_like(msk)), ("src", msk), ("bg", ~msk), ("tru", tru), ("zero", ~tru)]:
                rec[f"rank_pix_{k}"].append(torch.bincount(r[sel], minlength=N + 1).cpu().numpy())
                rec[f"rank_clip_{k}"].append(torch.bincount(rc[sel], minlength=N + 1).cpu().numpy())
            rec["mask_frac"].append([msk.float().mean().item(), tru.float().mean().item(), (msk & tru).float().sum().item() / max(tru.sum().item(), 1)])
            # 3a. patch-mean coverage (all patches; and patches with >= 50% source-mask pixels)
            pr, psr = [], []
            for s in PATCH:
                Sp, xp = patch_means(S, s), patch_means(x[None], s)[0]
                mp = patch_means(msk[None].float(), s)[0] >= 0.5
                rr = ranks(Sp, xp)
                pr.append(torch.bincount(rr, minlength=N + 1).cpu().numpy())
                psr.append(torch.bincount(rr[mp], minlength=N + 1).cpu().numpy())
            rec["patch_rank"].append(pr)
            rec["patch_src_rank"].append(psr)
            # 3b. power spectra
            Ps, Px, Pm = spec(S), spec(x[None])[0], spec(m[None])[0]
            Fm, Fx = torch.fft.fft2(m), torch.fft.fft2(x)
            cross = ((Fm * Fx.conj()).real / (H * H)).reshape(1, -1) @ spec.M
            rec["pow_samp"].append(Ps.mean(0).cpu().numpy())
            rec["pow_logsamp"].append(Ps.log().mean(0).cpu().numpy())
            rec["pow_true"].append(Px.cpu().numpy())
            rec["pow_mean"].append(Pm.cpu().numpy())
            rec["pow_cross"].append(cross[0].cpu().numpy())
            rec["pow_rank"].append(ranks(Ps, Px).cpu().numpy())
            # 3c. scores
            rec["es"].append(fair_energy_score(S[None, :, None], x[None, None]).item())
            rec["vs"].append(variogram(S, x))
            rec["crps"].append(fair_crps(S[None, :, None], x[None, None]).item())
            rec["mscrps"].append(ms_fair_crps(S[None, :, None], x[None, None]).item())
            # total flux: truth, sample mean, sample sd, clipped-sample mean
            fl = S.sum((1, 2))
            rec["flux"].append([x.sum().item(), fl.mean().item(), fl.std().item(), S.clamp_min(0).sum((1, 2)).mean().item()])
            # 4. diversity / collapse
            var = S.var(0)
            rec["v"].append(var.mean().item())
            rec["e"].append(((m - x) ** 2).mean().item())
            rec["v_pix_src"].append(var[msk].mean().item() if msk.any() else np.nan)
            rec["e_pix_src"].append(((m - x) ** 2)[msk].mean().item() if msk.any() else np.nan)
            Sf = S.reshape(N, -1).double()
            D = torch.cdist(Sf, Sf)
            iu = torch.triu_indices(N, N, 1, device=dev)
            pd = D[iu[0], iu[1]]
            td = (Sf - x.reshape(1, -1).double()).norm(dim=1)
            rec["pair_d"].append(pd.mean().item())
            rec["truth_d"].append(td.mean().item())
            rec["near_ident"].append([(pd < f * td.mean()).float().mean().item() for f in (0.01, 0.05, 0.1)])
            if i in FIG_IMAGES:
                figs[i] = dict(x=X[i], dirty=Y[i], psf=UV[i], samples=S[:8].cpu().numpy(), mean=m.cpu().numpy(),
                               std=S.std(0).cpu().numpy(), mask=msk.cpu().numpy())
            rec["time"].append(time.time() - t0)
            if i % 50 == 0:
                print(i, f"{np.mean(rec['time']):.2f}s/img psnr8 {ps[1]:.2f} v/e {rec['v'][-1] / rec['e'][-1]:.3f}", flush=True)
    out = {k: np.array(v) for k, v in rec.items()}
    out["k_eff"] = spec.k_eff
    out["n_samp"] = N
    np.savez_compressed(f"{a.out}/stats_{a.tag}.npz", **out)
    np.save(f"{a.out}/figs_{a.tag}.npy", figs, allow_pickle=True)
    print("done", a.tag, flush=True)


if __name__ == "__main__":
    main()
