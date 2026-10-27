"""Per-slice evaluation statistics from stored samples (GPU job, one method per call).

Inputs: <out>/samples/<name>/ from sample.py (magnitude images divided by the slice's GT max).
Output: <out>/stats/<name>.npz with per-slice arrays (leading axis = test slice), aggregated by aggregate.py.

Masks: `support` = ESPIRiT map support (outside it every sample and the truth are exactly 0, so coverage
is trivial there); `fg` = support AND (GT smoothed by a 5x5 box > 0.05), the tissue foreground used for
per-pixel calibration. Patch statistics use tiles with >= 75% of their pixels in fg (mean over fg pixels).
"""
import argparse
import json
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_scripts.metrics import psnr, ssim  # noqa: E402
from mri_crps.data import OUT  # noqa: E402
from utils.embeddings import VGG16Embedding  # noqa: E402

LEVELS = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
P_LIST = [1, 2, 4, 8, 32]
PATCH = [1, 4, 8, 16, 32, 64]
FG_THR = 0.05
N_RANK_BINS = 10
VS_LAGS = [(0, 1), (1, 0), (1, 1), (1, -1)]
N_KBINS = 16


def central_j(N, a):
    """Order-statistic central interval [X_(j), X_(N+1-j)] closest to level a; exact level (N+1-2j)/(N+1)."""
    j = int(round((N + 1) * (1 - a) / 2))
    j = min(max(j, 1), N // 2)
    return j, (N + 1 - 2 * j) / (N + 1)


def rand_rank(S, t, gen):
    """Rank of t among samples S (N, ...) along dim 0, ties broken uniformly at random: values in 0..N."""
    lt = (S < t).sum(0)
    eq = (S == t).sum(0)
    u = torch.rand(lt.shape, generator=gen, device=S.device)
    return lt + torch.floor(u * (eq + 1)).long()


def interval_cover(S_sorted, t, N):
    """Coverage indicator per level for central order-statistic intervals: (len(LEVELS), ...)."""
    out = []
    for a in LEVELS:
        j, _ = central_j(N, a)
        lo, hi = S_sorted[j - 1], S_sorted[N - j]
        out.append(((t >= lo) & (t <= hi)).float())
    return torch.stack(out)


def tile_info(fg, s):
    n = fg.shape[-1] // s
    cnt = F.avg_pool2d(fg[None, None].float(), s)[0, 0] * s * s
    valid = cnt >= 0.75 * s * s
    return cnt, valid


def patch_means(X, fg, s):
    """X (..., H, W) -> (..., n_valid_tiles): mean over fg pixels of each tile with >= 75% fg."""
    if s == 1:
        return X[..., fg]
    cnt, valid = tile_info(fg, s)
    sh = X.shape[:-2]
    Xs = F.avg_pool2d((X * fg).reshape(-1, 1, *X.shape[-2:]), s).reshape(*sh, cnt.shape[0], cnt.shape[1]) * s * s
    return (Xs / cnt.clamp_min(1))[..., valid]


def radial_bins(H, device):
    ky = torch.fft.fftfreq(H, device=device) * H
    k = torch.sqrt(ky[:, None] ** 2 + ky[None, :] ** 2)
    edges = torch.cat([torch.zeros(1, device=device), torch.logspace(0, np.log10(H / 2), N_KBINS, device=device)])
    idx = torch.bucketize(k, edges, right=True) - 1  # bin b: edges[b] <= k < edges[b+1]; k=0 -> bin 0
    idx[k >= H / 2] = -1
    return idx, edges


def band_power(X, idx):
    """X (..., H, W) real -> (..., N_KBINS) mean |FFT|^2 per radial bin (ortho FFT)."""
    P = torch.fft.fft2(X.double(), norm='ortho').abs() ** 2
    P = P.reshape(*X.shape[:-2], -1)
    flat = idx.reshape(-1)
    keep = flat >= 0
    out = torch.zeros(*X.shape[:-2], N_KBINS, device=X.device, dtype=P.dtype)
    out.index_add_(-1, flat[keep], P[..., keep])
    cnt = torch.bincount(flat[keep], minlength=N_KBINS).clamp_min(1)
    return out / cnt


def fair_es(V, v):
    V, v = V.double(), v.double()
    N = V.shape[0]
    D = torch.cdist(V, V)
    e = (V - v).norm(dim=1)
    return (e.mean() - D.sum() / (2 * N * (N - 1))).item(), D, e


def variogram(S, x, fg, p=0.5):
    """Bias-corrected variogram score (p=0.5, unit weights), mean over fg pixel pairs at each local lag."""
    N = S.shape[0]
    out = []
    H, W = x.shape
    for dy, dx in VS_LAGS:
        ys, ye = max(0, -dy), H - max(0, dy)
        xs, xe = max(0, -dx), W - max(0, dx)
        a_sl = (slice(ys, ye), slice(xs, xe))
        b_sl = (slice(ys + dy, ye + dy), slice(xs + dx, xe + dx))
        valid = fg[a_sl] & fg[b_sl]
        A = (S[(slice(None),) + a_sl] - S[(slice(None),) + b_sl]).abs()[:, valid].double() ** p
        t = (x[a_sl] - x[b_sl]).abs()[valid].double() ** p
        out.append((((t - A.mean(0)) ** 2) - A.var(0) / N).mean().item())
    return out


def rgb(im, unit_norm=False):
    """As scripts/mri/test.py: min-max to [-1, 1] (or [0, 1]), replicated to 3 channels. im (B, H, W)."""
    mn = im.amin((1, 2), keepdim=True)
    mx = im.amax((1, 2), keepdim=True)
    t = (im - mn) / (mx - mn)
    if not unit_norm:
        t = 2 * t - 1
    return t[:, None].expand(-1, 3, -1, -1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--n-use', type=int, default=None, help='use the first n samples (default all)')
    args = ap.parse_args()
    torch.set_float32_matmul_precision('highest')
    dev = 'cuda'
    src = OUT / 'samples' / args.name
    S_all = np.load(src / 'samples.npy', mmap_mode='r')
    GT = np.load(src / 'gt.npy')
    ZF = np.load(src / 'zf.npy')
    SUP = np.load(src / 'support.npy')
    n_items = S_all.shape[0]
    N = S_all.shape[1] if args.n_use is None else args.n_use
    lp = lpips.LPIPS(net='alex').to(dev)
    vgg = VGG16Embedding()
    imnet = (torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1),
             torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1))
    kidx, kedges = radial_bins(384, dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    box = torch.ones(1, 1, 5, 5, device=dev) / 25
    res = {}

    def put(k, v):
        res.setdefault(k, []).append(np.asarray(v))

    for i in range(n_items):
        S = torch.from_numpy(np.asarray(S_all[i, :N])).to(dev)  # (N, H, W)
        x = torch.from_numpy(GT[i]).to(dev)
        zf = torch.from_numpy(ZF[i]).to(dev)
        sup = torch.from_numpy(SUP[i]).to(dev)
        fg = sup & (F.conv2d(x[None, None], box, padding=2)[0, 0] > FG_THR)
        put('n_fg', fg.sum().item())
        put('n_sup', sup.sum().item())
        gt_np = GT[i]

        # 1. reconstruction metrics of the P-sample mean (repo definitions)
        for P in P_LIST:
            m = S[:P].mean(0)
            m_np = m.cpu().numpy()
            put(f'psnr_P{P}', psnr(gt_np, m_np))
            put(f'ssim_P{P}', ssim(gt_np, m_np))
            with torch.no_grad():
                put(f'lpips_P{P}', lp(rgb(x[None]), rgb(m[None])).item())
        put('apsd32', S[:32].std(0).mean().item())  # repo's APSD (whole image, 32 samples)
        put('psnr_zf', psnr(gt_np, ZF[i]))

        # CFID embeddings: first 32 samples, condition = zero-filled magnitude (repo: VGG16 avg-pooled features)
        with torch.no_grad():
            def emb(t):
                return vgg((rgb(t, unit_norm=True) - imnet[0]) / imnet[1]).cpu().numpy()
            put('emb_gen', emb(S[:32]))
            put('emb_true', emb(x[None])[0])
            put('emb_cond', emb(zf[None])[0])

        # 2. per-pixel calibration in fg
        Sf = S[:, fg]
        tf = x[fg]
        Ss, _ = Sf.sort(0)
        cov = interval_cover(Ss, tf, N)  # (L, n_fg)
        put('pix_cov', cov.mean(1).cpu().numpy())
        r = rand_rank(Sf, tf, gen)
        put('pix_rankhist', torch.bincount((r * N_RANK_BINS) // (N + 1), minlength=N_RANK_BINS).cpu().numpy())
        # pixels in support but outside fg (background noise region)
        bg = sup & ~fg
        if bg.any():
            covb = interval_cover(S[:, bg].sort(0)[0], x[bg], N)
            put('bg_cov', covb.mean(1).cpu().numpy())
        else:
            put('bg_cov', np.full(len(LEVELS), np.nan))

        # 3a. patch-mean coverage / ranks
        for s in PATCH[1:]:
            Ps = patch_means(S, fg, s)
            pt = patch_means(x, fg, s)
            if pt.numel() == 0:
                put(f'patch{s}_cov', np.full(len(LEVELS), np.nan))
                put(f'patch{s}_rankhist', np.zeros(N_RANK_BINS, int))
                put(f'patch{s}_n', 0)
                continue
            put(f'patch{s}_cov', interval_cover(Ps.sort(0)[0], pt, N).mean(1).cpu().numpy())
            rr = rand_rank(Ps, pt, gen)
            put(f'patch{s}_rankhist', torch.bincount((rr * N_RANK_BINS) // (N + 1), minlength=N_RANK_BINS).cpu().numpy())
            put(f'patch{s}_n', pt.numel())
            # spread-skill at patch level
            pm = Ps.mean(0)
            put(f'patch{s}_err2', ((pm - pt) ** 2).sum().item())
            put(f'patch{s}_var', Ps.var(0).sum().item())
        g_s = (S * fg).sum((1, 2)) / fg.sum()
        g_t = (x * fg).sum() / fg.sum()
        put('global_rank', rand_rank(g_s[:, None], g_t[None], gen)[0].item())
        put('global_cov', interval_cover(g_s.sort(0)[0][:, None], g_t[None], N)[:, 0].cpu().numpy())

        # 3b. radially averaged power spectra (whole 384x384 magnitude image)
        bp_s = band_power(S, kidx)  # (N, K)
        bp_x = band_power(x, kidx)
        m = S.mean(0)
        put('bp_s_mean', bp_s.mean(0).cpu().numpy())
        put('bp_x', bp_x.cpu().numpy())
        put('bp_m', band_power(m, kidx).cpu().numpy())
        put('bp_res_x', band_power(x - m, kidx).cpu().numpy())        # error of the posterior mean
        put('bp_res_s', band_power(S - m, kidx).mean(0).cpu().numpy())  # spread around the mean
        put('bp_rank', rand_rank(bp_s, bp_x, gen).cpu().numpy())

        # 3c. fair energy score (whole image vector, fg pixels) and variogram score
        es, D, e = fair_es(Sf, tf)
        put('es_fair', es)
        put('vs_local', variogram(S, x, fg))

        # 4. diversity / spread-skill (fg pixels)
        put('err2_mean', ((m[fg] - tf) ** 2).sum().item())
        put('var_sum', Sf.var(0).sum().item())
        off = ~torch.eye(N, dtype=bool, device=dev)
        dbar = D[off].mean()
        put('pair_dbar', dbar.item())
        put('err_mean_norm', (m[fg] - tf).norm().item())
        put('frac_pairs_lt_half_mean', (D[off] < 0.5 * dbar).float().mean().item())
        put('min_pair_over_mean', (D[off].min() / dbar).item())
        put('err_samp_mean', e.mean().item())
        if i % 10 == 0:
            print(f'{args.name}: {i + 1}/{n_items}', flush=True)

    out = {k: np.stack(v) for k, v in res.items()}
    (OUT / 'stats').mkdir(exist_ok=True)
    np.savez(OUT / 'stats' / f'{args.name}.npz', N=N, levels=np.array(LEVELS),
             exact=np.array([central_j(N, a)[1] for a in LEVELS]), kedges=kedges.cpu().numpy(), **out)
    print('saved', OUT / 'stats' / f'{args.name}.npz')
