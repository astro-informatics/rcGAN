"""Coverage suite on the ORIGINAL RIGAN models (Mars et al. 2025, arXiv:2507.21270; Zenodo 16637065 v2: checkpoints + test set).

GPU pass: N posterior samples per test image (torch.manual_seed(1000 + i), same chunking for both models -> paired),
per-image statistics streamed to an .npz; aggregation/bootstrap in orig_analyse.py. Sky units = x as stored
(each truth image normalised to peak 1); samples de-normalised with the 'macro' train mean/std shipped on Zenodo.

Masks (y-based ones use only the dirty image, so they are the same for every model):
  all     all 360x360 pixels
  src     Gaussian-smoothed (sigma 2 px) dirty > 0.1 max(dirty), dilated 2 px  (as in the paper runs)
  bg      complement of src
  bright  truth >= 0.1 (truth-based; diagnostic only)

Usage: python orig_eval.py --model unet|gunet [--n-test 1900 --n-samp 128 --chunk 16]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/home/jjwhit/.herdr/worktrees/rcGAN/radio-orig-eval"
sys.path.insert(0, ROOT)
from models.lightning.riGAN import riGAN  # noqa: E402
from models.lightning.GriGAN import GriGAN  # noqa: E402

BASE = "/share/gpu0/jjwhit/crps_review/radio_orig/"
DATA = BASE + "data/test/"
NORM = BASE + "models/zenodo_RIGAN/train/"
CKPT = {"unet": BASE + "models/zenodo_RIGAN/radio_30dor/checkpoint_best.ckpt",
        "gunet": BASE + "models/zenodo_RIGAN/radio_30dor_gradient/checkpoint_best.ckpt"}
H = 360
PATCH = [2, 4, 8, 16, 32, 64, 360]  # k x k tiles over the top-left floor(360/k)*k crop; 360 = whole image
MASKS = ["all", "src", "bg", "bright"]
SKEW_EDGES = np.linspace(-6, 6, 121)
KURT_EDGES = np.linspace(-3, 30, 166)
Z_EDGES = np.linspace(-8, 8, 161)
SD_EDGES = np.logspace(-5, 0, 26)  # spread-skill reliability bins in posterior sd (sky units)
FIG_IMAGES = [0, 1, 2, 3]


def gauss_smooth(img, sigma):
    f = torch.fft.fftfreq(H, device=img.device)
    k = torch.exp(-2 * np.pi ** 2 * (f[:, None] ** 2 + f[None, :] ** 2) * sigma ** 2)
    return torch.fft.ifft2(torch.fft.fft2(img) * k).real


def source_mask(dirty):
    m = (gauss_smooth(dirty, 2.0) > 0.1 * dirty.max()).float()
    return F.max_pool2d(m[None, None], 5, stride=1, padding=2)[0, 0] > 0


def tiles_mean(X, k):
    n = H // k
    X = X[..., : n * k, : n * k]
    return X.reshape(*X.shape[:-2], n, k, n, k).mean((-3, -1)).reshape(*X.shape[:-2], -1)


def ssim_psnr(pred, gt):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    p, g = pred.astype(np.float64), gt.astype(np.float64)
    return (peak_signal_noise_ratio(g, p, data_range=g.max() - g.min()),
            structural_similarity(g, p, data_range=g.max() - g.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CKPT))
    ap.add_argument("--n-test", type=int, default=1900)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-samp", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--out", default=BASE + "eval/")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    cls = GriGAN if a.model == "gunet" else riGAN
    model = cls.load_from_checkpoint(CKPT[a.model], map_location="cpu").to(dev).eval()
    print(a.model, "std_mult", getattr(model, "std_mult", None), "im_size", model.resolution, flush=True)
    mx, sx, my, sy, mu, su = [float(np.load(NORM + f"{k}.npy")) for k in ["mean_x", "std_x", "mean_y", "std_y", "mean_uv", "std_uv"]]
    X = np.load(DATA + "x.npy", mmap_mode="r")
    Y = np.load(DATA + "y.npy", mmap_mode="r")
    U = np.load(DATA + "uv.npy", mmap_mode="r")
    idx = range(a.start, min(a.start + a.n_test, len(X)))
    N = a.n_samp
    nb = len(SD_EDGES) - 1

    rec = {k: [] for k in ["img", "psnr", "ssim", "psnr32", "ssim32", "snr32", "snrN", "corr_std_err32", "corr_std_errN",
                           "mse", "var", "mse_src", "var_src", "rank_img", "neg_frac_src", "neg_frac_bg", "zero_true_frac"]}
    for m in MASKS:
        for k in ["rank", "gpit"]:
            rec[f"{k}_{m}"] = []
        rec[f"skew_{m}"], rec[f"kurt_{m}"], rec[f"z_{m}"] = [], [], []
    for k in PATCH[:-1]:
        rec[f"prank_{k}"] = []
    ss_var = np.zeros(nb)
    ss_err = np.zeros(nb)
    ss_n = np.zeros(nb)
    figs = {}
    t0 = time.time()
    for i in idx:
        x = torch.from_numpy(np.array(X[i])).float().to(dev)
        d = torch.from_numpy(np.array(Y[i])).float().to(dev)
        psf = torch.from_numpy(np.array(U[i])).float().to(dev)
        y = torch.stack([(d - my) / sy, (psf - mu) / su])[None]
        with torch.no_grad():
            torch.manual_seed(1000 + i)
            S = torch.cat([model(y.repeat(min(a.chunk, N - j), 1, 1, 1)) for j in range(0, N, a.chunk)])[:, 0]
        S = (S * sx + mx).double()
        xd = x.double()
        src = source_mask(d)
        masks = {"all": torch.ones_like(src), "src": src, "bg": ~src, "bright": xd >= 0.1}
        mean = S.mean(0)
        sd = S.std(0)
        mean32 = S[:32].mean(0)
        sd32 = S[:32].std(0)
        # 1. accuracy (posterior mean) + paper metrics (SNR, Pearson(std, |err|) with 32 samples)
        pn, sn = ssim_psnr(mean.cpu().numpy(), x.cpu().numpy())
        p32, s32 = ssim_psnr(mean32.cpu().numpy(), x.cpu().numpy())
        rec["psnr"].append(pn); rec["ssim"].append(sn); rec["psnr32"].append(p32); rec["ssim32"].append(s32)
        rec["snr32"].append((20 * torch.log10(xd.norm() / (xd - mean32).norm())).item())
        rec["snrN"].append((20 * torch.log10(xd.norm() / (xd - mean).norm())).item())
        rec["corr_std_err32"].append(np.corrcoef(sd32.flatten().cpu().numpy(), (xd - mean32).abs().flatten().cpu().numpy())[0, 1])
        rec["corr_std_errN"].append(np.corrcoef(sd.flatten().cpu().numpy(), (xd - mean).abs().flatten().cpu().numpy())[0, 1])
        # 2. spread-skill (per image; per-pixel reliability bins)
        err2 = (xd - mean) ** 2
        var = sd ** 2
        rec["mse"].append(err2.mean().item()); rec["var"].append(var.mean().item())
        rec["mse_src"].append(err2[src].mean().item()); rec["var_src"].append(var[src].mean().item())
        b = np.clip(np.digitize(sd.flatten().cpu().numpy(), SD_EDGES) - 1, 0, nb - 1)
        ss_var += np.bincount(b, var.flatten().cpu().numpy(), nb)
        ss_err += np.bincount(b, err2.flatten().cpu().numpy(), nb)
        ss_n += np.bincount(b, minlength=nb)
        # 3. per-pixel ranks (continuous truth and samples: no ties) + Gaussian PIT + shape of the per-pixel posterior
        r = (S < xd).sum(0)
        z = (xd - mean) / sd
        gp = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
        c = S - mean
        m2 = (c ** 2).mean(0)
        skew = (c ** 3).mean(0) / m2 ** 1.5
        kurt = (c ** 4).mean(0) / m2 ** 2 - 3
        for m, msk in masks.items():
            rec[f"rank_{m}"].append(torch.bincount(r[msk], minlength=N + 1).cpu().numpy())
            rec[f"gpit_{m}"].append(np.histogram(gp[msk].cpu().numpy(), bins=N + 1, range=(0, 1))[0])
            rec[f"skew_{m}"].append(np.histogram(skew[msk].cpu().numpy(), SKEW_EDGES)[0])
            rec[f"kurt_{m}"].append(np.histogram(kurt[msk].cpu().numpy(), KURT_EDGES)[0])
            rec[f"z_{m}"].append(np.histogram(z[msk].cpu().numpy(), Z_EDGES)[0])
        rec["neg_frac_src"].append((S[:, src] < 0).double().mean().item())
        rec["neg_frac_bg"].append((S[:, ~src] < 0).double().mean().item())
        rec["zero_true_frac"].append((xd < 1e-3).double().mean().item())
        # 4. patch means (scale-dependent coverage); whole image = total flux
        for k in PATCH:
            pr = (tiles_mean(S, k) < tiles_mean(xd[None], k)[0]).sum(0)
            if k == H:
                rec["rank_img"].append(int(pr[0]))
            else:
                rec[f"prank_{k}"].append(torch.bincount(pr, minlength=N + 1).cpu().numpy())
        rec["img"].append(i)
        if i in FIG_IMAGES:
            figs[i] = dict(x=x.cpu().numpy(), dirty=d.cpu().numpy(), mean=mean.float().cpu().numpy(),
                           sd=sd.float().cpu().numpy(), samples=S[:4].float().cpu().numpy())
        if (i - a.start) % 50 == 0:
            print(a.model, i, f"{time.time() - t0:.0f}s psnr {pn:.2f} snr32 {rec['snr32'][-1]:.2f}", flush=True)
    out = {k: np.array(v) for k, v in rec.items()}
    out.update(ss_var=ss_var, ss_err=ss_err, ss_n=ss_n, sd_edges=SD_EDGES, skew_edges=SKEW_EDGES, kurt_edges=KURT_EDGES,
               z_edges=Z_EDGES, n_samp=N, patch=np.array(PATCH))
    for i, f in figs.items():
        for k, v in f.items():
            out[f"fig{i}_{k}"] = v
    tag = f"{a.model}" + (f"_{a.start}" if a.start else "")
    np.savez_compressed(a.out + f"stats_{tag}.npz", **out)
    print("done", a.model, len(out["img"]), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
