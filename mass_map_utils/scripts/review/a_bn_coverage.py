"""Side check: A1 coverage with train-mode BatchNorm vs eval mode (R and F3).

Train-mode BN normalises with batch statistics. The requested BN batch is "the N
samples of one map"; the 16 GB GPUs cannot hold N=256 at 300x300 in one forward,
so the batch is a chunk of samples of the same map. BN statistics are per-channel
means over (batch x H x W) with H x W = 90000, so they are dominated by spatial
averaging; the sensitivity to the chunk size is measured directly (--chunks).

Usage: python a_bn_coverage.py --models R F3 --n-maps 100 --n-samp 256 --chunks 8 2
"""
import argparse
import json
import os

import numpy as np
import torch

from common import MODELS, OUT_ROOT, TestSet, load_cfg, load_model, sample

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["R", "F3"])
p.add_argument("--n-maps", type=int, default=100)
p.add_argument("--n-samp", type=int, default=256)
p.add_argument("--chunks", type=int, nargs="+", default=[8, 2])
p.add_argument("--n-maps-chunk-check", type=int, default=10)
args = p.parse_args()

torch.set_float32_matmul_precision("medium")
cfg = load_cfg()
ts = TestSet(cfg)
mask = torch.from_numpy(ts.mask).cuda()
N = args.n_samp
LEVELS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def coverage(rank):
    rows = []
    for L in LEVELS:
        k = max(1, int(round((N + 1) * (1 - L) / 2)))
        rows.append(((N + 1 - 2 * k) / (N + 1), ((rank >= k) & (rank <= N - k)).float().mean().item()))
    return rows


def run(model, mode, chunk, n_maps):
    model.train() if mode == "train" else model.eval()
    covs, sds = [], []
    for i in range(n_maps):
        y, x = ts.get(i)
        S = sample(model, y, N, chunk=chunk, seed=i)[:, mask]
        t = (x[0].cuda() * cfg.kappa_std + cfg.kappa_mean)[mask]
        covs.append([c for _, c in coverage((S < t).sum(0))])
        sds.append(S.std(0).mean().item())
    model.eval()
    exact = [e for e, _ in coverage(torch.zeros(1, device="cuda"))]
    return {"exact": exact, "cov": np.mean(covs, 0).tolist(), "cov_se": (np.std(covs, 0, ddof=1) / np.sqrt(n_maps)).tolist(),
            "mean_sd": float(np.mean(sds))}


res = {}
for key in args.models:
    model = load_model(MODELS[key])  # fresh: running stats untouched by train-mode passes below
    r = {"eval": run(model, "eval", 32, args.n_maps)}
    # train-mode passes update running stats; they are never used again (eval runs above, model discarded after)
    r[f"train_chunk{args.chunks[0]}"] = run(model, "train", args.chunks[0], args.n_maps)
    for c in args.chunks[1:]:
        r[f"train_chunk{c}_first{args.n_maps_chunk_check}"] = run(model, "train", c, args.n_maps_chunk_check)
    r[f"train_chunk{args.chunks[0]}_first{args.n_maps_chunk_check}"] = run(model, "train", args.chunks[0], args.n_maps_chunk_check)
    res[key] = r
    print(key, json.dumps({k: (v["mean_sd"], [round(v["cov"][j] - v["exact"][j], 4) for j in (9, 17)]) for k, v in r.items()}), flush=True)
    del model
    torch.cuda.empty_cache()

os.makedirs(OUT_ROOT + "a0", exist_ok=True)
json.dump(res, open(OUT_ROOT + "a0/bn_coverage.json", "w"), indent=1)
