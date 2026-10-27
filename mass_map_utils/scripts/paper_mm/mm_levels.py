"""Per-level decomposition of the multiscale fair CRPS (diagnostic).
For each model, draw P samples for n test maps (stage-A TestSet, latent seed = map index), and compute in the
*training* units (normalised kappa, full 300x300 field, exactly the tensor the loss sees) the fair CRPS of the
orthonormal Haar scaling coefficients at each level s = 0..6 (proper_scores.ms_fair_crps internals), plus its
two terms E|g - x| and 0.5 E|g - g'|. Writes EVAL/levels/<key>.json.
Usage: python mm_run.py stageA $PWD/mm_levels.py --model MS [--ckpt ...] [--n-maps 100] [--P 8]
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

import common
from common import TestSet, load_cfg
from models.lightning.proper_scores import _mean_abs_pairs, _mean_abs_to_truth, default_levels

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--ckpt", default="checkpoint_best.ckpt")
p.add_argument("--n-maps", type=int, default=100)
p.add_argument("--P", type=int, default=8)
args = p.parse_args()
cfg = load_cfg()
ts = TestSet(cfg)
model = common.load_model(common.MODELS[args.model], ckpt=args.ckpt)
L = default_levels(300, 300)
acc = torch.zeros(L, 3, dtype=torch.float64)
with torch.no_grad():
    for i in range(args.n_maps):
        y, x = ts.get(i)
        torch.manual_seed(i)
        g = model(y[None].cuda().expand(args.P, -1, -1, -1).contiguous())[None].double()  # (1, P, 1, H, W) normalised
        t = x[None].cuda().double()
        gg = g.reshape(args.P, 1, 300, 300)
        for s in range(L):
            if s > 0:
                gg = 2.0 * F.avg_pool2d(gg, 2, ceil_mode=True)
                t = 2.0 * F.avg_pool2d(t, 2, ceil_mode=True)
            G = gg.view(1, args.P, 1, gg.shape[-2], gg.shape[-1])
            a = _mean_abs_to_truth(G, t).mean().item()
            b = 0.5 * _mean_abs_pairs(G, include_diagonal=False).mean().item()
            acc[s] += torch.tensor([a - b, a, b], dtype=torch.float64)
acc /= args.n_maps
res = {"model": args.model, "ckpt": args.ckpt, "n_maps": args.n_maps, "P": args.P, "levels": L,
       "block_px": [2 ** s for s in range(L)], "fair_crps": acc[:, 0].tolist(), "E|g-x|": acc[:, 1].tolist(),
       "0.5E|g-g'|": acc[:, 2].tolist(), "ms_fair_crps": acc[:, 0].mean().item(),
       "share_of_ms": (acc[:, 0] / acc[:, 0].sum()).tolist()}
d = common.OUT_ROOT.replace("stageA/", "levels/")
os.makedirs(d, exist_ok=True)
json.dump(res, open(os.path.join(d, f"{args.model}_{args.ckpt.replace('.ckpt', '')}.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
