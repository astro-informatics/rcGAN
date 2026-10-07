"""Stage A0 sanity checks (GPU).

a) BatchNorm: per-pixel sample SD with model.eval() (running BN stats, as plot.py)
   vs train-mode BN (batch statistics over a batch of 9 different maps, one
   forward call per sample, as in training_step), no grad.
c) Throughput: time to draw N samples for one map in eval mode.

Usage: python a0_sanity.py --models R F3 --n-maps 50 --n-samp 32
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import MODELS, OUT_ROOT, TestSet, load_cfg, load_model, sample, to_kappa

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["R", "F3"])
p.add_argument("--n-maps", type=int, default=50)
p.add_argument("--n-samp", type=int, default=32)
p.add_argument("--train-batch", type=int, default=9)
args = p.parse_args()

torch.set_float32_matmul_precision("medium")
cfg = load_cfg()
ts = TestSet(cfg)
mask = torch.from_numpy(ts.mask).cuda()
idx = list(range(args.n_maps))
results = {}

for key in args.models:
    exp = MODELS[key]
    model = load_model(exp)
    res = {}

    # --- eval-mode SD (independent samples, running BN stats)
    sd_eval = []
    for i in idx:
        y, _ = ts.get(i)
        s = sample(model, y, args.n_samp, chunk=32, seed=i)
        sd_eval.append(s.std(0, unbiased=True)[mask].mean().item())

    # --- train-mode BN: batch of `train_batch` different maps per forward call.
    # Running stats get updated in train mode, so this runs after eval mode and
    # the model is discarded afterwards.
    model.train()
    sd_train = []
    with torch.no_grad():
        torch.manual_seed(0)
        for b in range(0, len(idx), args.train_batch):
            bi = idx[b:b + args.train_batch]
            y, _ = ts.batch(bi)
            y = y.cuda()
            gens = torch.stack([model(y)[:, 0] for _ in range(args.n_samp)], 1)  # (B, N, H, W)
            gens = to_kappa(gens, cfg)
            sd = gens.std(1, unbiased=True)
            sd_train += [sd[j][mask].mean().item() for j in range(len(bi))]
    model.eval()

    sd_eval, sd_train = np.array(sd_eval), np.array(sd_train)
    res["bn_mean_sd_eval"] = float(sd_eval.mean())
    res["bn_mean_sd_train"] = float(sd_train.mean())
    res["bn_ratio_train_over_eval"] = float(sd_train.mean() / sd_eval.mean())
    res["bn_per_map_ratio_median"] = float(np.median(sd_train / sd_eval))

    # --- throughput (eval mode), reload to get clean running stats
    del model
    torch.cuda.empty_cache()
    model = load_model(exp)
    y, _ = ts.get(0)
    for chunk in [32, 64, 128]:
        try:
            sample(model, y, chunk, chunk=chunk)  # warm-up
            torch.cuda.synchronize()
            t0 = time.time()
            sample(model, y, 512, chunk=chunk)
            torch.cuda.synchronize()
            res[f"sec_per_512_chunk{chunk}"] = time.time() - t0
        except RuntimeError as e:  # OOM
            res[f"sec_per_512_chunk{chunk}"] = f"failed: {str(e)[:80]}"
            torch.cuda.empty_cache()
    res["max_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    results[key] = res
    print(key, exp, json.dumps(res, indent=1), flush=True)
    del model
    torch.cuda.empty_cache()

os.makedirs(OUT_ROOT + "a0", exist_ok=True)
with open(OUT_ROOT + "a0/a0_sanity.json", "w") as f:
    json.dump(results, f, indent=1)
