"""Stage A main pass (GPU): draw N posterior samples per test map and stream per-pixel
statistics over the COSMOS footprint (native orientation; see common.py). Samples
are not stored.

Per model, writes OUT_ROOT/stats/<key>/:
  pix_<name>.npy  (n_maps, P) memmaps over footprint pixels, P = mask.sum():
     m     sample mean                       float32
     s     sample SD (ddof=1)                float32
     skew  sample skewness (g1)              float16
     kurt  sample excess kurtosis (g2)       float16
     x     truth                             float32
     rank  #{samples < truth}, 0..N          uint16
     mae   mean_i |X_i - x|                  float32
     gmd   mean_{i!=j} |X_i - X_j|           float32   (fair CRPS = mae - gmd/2)
  maps.npz: per-map scalars
     es_fair      fair energy score of the footprint vector (Euclidean)
     ecp32        (n_maps, 19) old-style coverage from the first 32 samples, np.percentile
                  central intervals at levels 0.05..0.95 (as calibration.py, lam=1)
     rho_hist     (n_maps, n_levels, n_bins) histogram of |x - mean| / half-width, where
                  half-width = (q_hi - q_lo)/2 of the N samples at the level -> RCPS losses
     half_width   (n_maps, n_levels) mean half-width over footprint pixels
  meta.json

Usage: python a_sample_stats.py --model F3 --n-samp 512 [--n-maps 1000]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import MODELS, OUT_ROOT, TestSet, load_cfg, load_model, sample

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, choices=list(MODELS))
p.add_argument("--n-samp", type=int, default=512)
p.add_argument("--n-maps", type=int, default=1000)
p.add_argument("--chunk", type=int, default=64)
p.add_argument("--ckpt", default="checkpoint_best.ckpt")
p.add_argument("--stats-dir", default="stats", help="subdirectory of OUT_ROOT (use e.g. smoke for tests)")
args = p.parse_args()

torch.set_float32_matmul_precision("medium")
cfg = load_cfg()
ts = TestSet(cfg)
n_maps, N = min(args.n_maps, len(ts)), args.n_samp
mask = torch.from_numpy(ts.mask).cuda()
P = int(mask.sum())

RCPS_LEVELS = [0.5, 0.8, 0.9, 0.95]
RHO_EDGES = np.concatenate([np.linspace(0, 5, 2001), [np.inf]])  # lam grid step 0.0025
ECP_LEVELS = np.round(np.arange(0.05, 1.0, 0.05), 2)

out = os.path.join(OUT_ROOT, args.stats_dir, args.model)
os.makedirs(out, exist_ok=True)
spec = {"m": np.float32, "s": np.float32, "skew": np.float16, "kurt": np.float16, "x": np.float32,
        "rank": np.uint16, "mae": np.float32, "gmd": np.float32}
arrs = {k: np.lib.format.open_memmap(f"{out}/pix_{k}.npy", mode="w+", dtype=dt, shape=(n_maps, P))
        for k, dt in spec.items()}
es_fair = np.zeros(n_maps)
ecp32 = np.zeros((n_maps, len(ECP_LEVELS)))
rho_hist = np.zeros((n_maps, len(RCPS_LEVELS), len(RHO_EDGES) - 1), dtype=np.int32)
half_width = np.zeros((n_maps, len(RCPS_LEVELS)))

def sorted_quantile(srt, q):
    """np.percentile 'linear' quantile along dim 0 of an already-sorted tensor
    (torch.quantile in torch 1.13 rejects inputs > 2^24 elements)."""
    pos = q * (srt.shape[0] - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, srt.shape[0] - 1)
    return srt[lo] + (pos - lo) * (srt[hi] - srt[lo])


model = load_model(MODELS[args.model], ckpt=args.ckpt)
k_idx = torch.arange(1, N + 1, device="cuda", dtype=torch.float64)
gmd_w = (2 * k_idx - N - 1)[:, None]  # sum_{i<j}(X_(j)-X_(i)) = sum_k (2k-N-1) X_(k)
t0 = time.time()

for i in range(n_maps):
    y, x = ts.get(i)
    S = sample(model, y, N, chunk=args.chunk, seed=i)  # (N, H, W), kappa units
    S = S[:, mask]  # (N, P)
    t = (x[0].cuda() * cfg.kappa_std + cfg.kappa_mean)[mask]  # (P,)

    Sd = S.double()
    m = Sd.mean(0)
    c = Sd - m
    var_b = (c ** 2).mean(0)
    s = (var_b * N / (N - 1)).sqrt()
    skew = (c ** 3).mean(0) / var_b ** 1.5
    kurt = (c ** 4).mean(0) / var_b ** 2 - 3
    srt, _ = torch.sort(Sd, 0)
    gmd = 2 * (gmd_w * srt).sum(0) / (N * (N - 1))
    mae = (Sd - t.double()).abs().mean(0)
    rank = (S < t).sum(0)

    for k, v in zip(["m", "s", "skew", "kurt", "x", "rank", "mae", "gmd"], [m, s, skew, kurt, t, rank, mae, gmd]):
        arrs[k][i] = v.cpu().numpy().astype(spec[k])

    # fair energy score of the whole footprint vector
    # pairwise distances from a float64 Gram matrix of centred samples (cdist would use TF32 matmul)
    G = c @ c.T
    dg = G.diagonal()
    D = (dg[:, None] + dg[None, :] - 2 * G).clamp_min(0).sqrt()
    D.fill_diagonal_(0)
    e1 = (Sd - t.double()).norm(dim=1).mean()
    es_fair[i] = (e1 - D.sum() / (2 * N * (N - 1))).item()

    # old 32-sample percentile ECP (np.percentile linear interpolation)
    s32, _ = torch.sort(Sd[:32], 0)
    for li, L in enumerate(ECP_LEVELS):
        lo, hi = sorted_quantile(s32, (1 - L) / 2), sorted_quantile(s32, (1 + L) / 2)
        ecp32[i, li] = ((t >= lo) & (t <= hi)).double().mean().item()

    # RCPS: rho = |x - mean| / half-width, covered at lam iff rho <= lam
    for li, L in enumerate(RCPS_LEVELS):
        h = (sorted_quantile(srt, (1 + L) / 2) - sorted_quantile(srt, (1 - L) / 2)) / 2
        rho = ((t.double() - m).abs() / h).cpu().numpy()
        rho_hist[i, li] = np.histogram(rho, bins=RHO_EDGES)[0]
        half_width[i, li] = h.mean().item()

    if i % 50 == 0 or i == n_maps - 1:
        print(f"[{args.model}] map {i + 1}/{n_maps}  {time.time() - t0:.0f}s", flush=True)

for a in arrs.values():
    a.flush()
np.savez(f"{out}/maps.npz", es_fair=es_fair, ecp32=ecp32, ecp_levels=ECP_LEVELS, rho_hist=rho_hist,
         rho_edges=RHO_EDGES, rcps_levels=RCPS_LEVELS, half_width=half_width)
json.dump({"model": args.model, "exp": MODELS[args.model], "ckpt": args.ckpt, "n_samp": N, "n_maps": n_maps,
           "P": P, "noise_seed": "1_000_000 + map index (sorted kappa_test)", "latent_seed": "map index",
           "orientation": "native generator (B,1,H,W); no reformat/rotation; COSMOS mask applied directly",
           "seconds": time.time() - t0}, open(f"{out}/meta.json", "w"), indent=1)
print("done", time.time() - t0)
