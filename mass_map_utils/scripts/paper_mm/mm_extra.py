"""Extra: transfer slope of the posterior mean, T(ell) = sum P_{m,x} / sum P_x (masked spectra),
i.e. the regression slope of posterior-mean Fourier modes on truth modes (1 = unbiased amplitude,
< 1 = shrinkage). Bootstrap over maps (B = 2000, seed 0), paired differences vs R.
Usage: python jt_extra.py
"""
import json
import os

import numpy as np

import sys
OUT = "/share/gpu0/jjwhit/crps_review/paper/mm/eval/joint/"
MOD = sys.argv[1:] or ["R", "F100", "MS"]
d = {k: (np.load(f"{OUT}main/{k}/bp_mask_cross_mx.npy"), np.load(f"{OUT}main/{k}/bp_mask_x.npy")) for k in MOD}
k_eff = np.array(json.load(open(f"{OUT}main/R/meta.json"))["k_eff"])
ell = k_eff * 2 * np.pi / np.deg2rad(300 * 0.29 / 60)
n = d["R"][0].shape[0]
rng = np.random.default_rng(0)
C = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(2000)]).astype(float)
res = {"ell": ell.round(0).tolist()}
draws = {}
for k in MOD:
    cx, px = d[k]
    res[k] = (cx.sum(0) / px.sum(0)).round(4).tolist()
    draws[k] = (C @ cx) / (C @ px)
    res[k + "_ci"] = np.percentile(draws[k], [2.5, 97.5], 0).round(4).tolist()
for k in MOD[1:]:
    if k != "F100" and "F100" in MOD:
        res[k + "_minus_F100_ci"] = np.percentile(draws[k] - draws["F100"], [2.5, 97.5], 0).round(4).tolist()
    res[k + "_minus_R_ci"] = np.percentile(draws[k] - draws["R"], [2.5, 97.5], 0).round(4).tolist()
json.dump(res, open(f"{OUT}analysis/transfer.json", "w"), indent=1)
for k in MOD:
    print(k, " ".join(f"{e:.0f}:{v:.3f}" for e, v in zip(ell, res[k])))
