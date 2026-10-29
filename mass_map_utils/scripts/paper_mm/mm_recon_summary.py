"""Summarise mm_recon.py outputs: mean over maps of PSNR/SNR/RMSE/Pearson/SSIM of the P-sample mean,
P in {1, 8, 32}, with bootstrap 95% CIs (B = 2000 over maps, seed 0, shared indices) and paired
differences vs R and vs F100. Writes EVAL/recon/recon_summary.json and prints a markdown table.
Usage: python mm_recon_summary.py [--tag seeds] R F100 MS ...   (--tag X writes recon_summary_X.json)
"""
import json
import os
import sys

import numpy as np

D = "/share/gpu0/jjwhit/crps_review/paper/mm/eval/recon/"
ARGS = sys.argv[1:]
TAG = ""
if ARGS[:1] == ["--tag"]:
    TAG, ARGS = "_" + ARGS[1], ARGS[2:]
MOD = [k for k in ARGS if os.path.exists(D + k + ".npz")]
Z = {k: np.load(D + k + ".npz") for k in MOD}
n = min(int(Z[k]["n_maps"]) for k in MOD)
rng = np.random.default_rng(0)
C = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(2000)]).astype(float) / n
P = Z[MOD[0]]["P"].tolist()
out = {"n_maps": n, "P": P}
rows = []
for met in ["psnr", "ssim", "rmse", "pearson", "snr"]:
    out[met] = {}
    draws = {k: C @ Z[k][met][:n] for k in MOD}
    for k in MOD:
        v = Z[k][met][:n].mean(0)
        e = {"val": v.tolist(), "ci": np.percentile(draws[k], [2.5, 97.5], 0).T.tolist()}
        for ref in ["R", "F100"]:
            if ref in draws and k != ref:
                e[f"minus_{ref}"] = (v - Z[ref][met][:n].mean(0)).tolist()
                e[f"minus_{ref}_ci"] = np.percentile(draws[k] - draws[ref], [2.5, 97.5], 0).T.tolist()
        out[met][k] = e
        cells = []
        for j in range(len(P)):
            s = f"{v[j]:.4g} [{e['ci'][j][0]:.4g}, {e['ci'][j][1]:.4g}]"
            if "minus_R" in e:
                s += f"; −R {e['minus_R'][j]:+.3g} [{e['minus_R_ci'][j][0]:+.3g}, {e['minus_R_ci'][j][1]:+.3g}]"
            cells.append(s)
        rows.append(f"| {met} | {k} | " + " | ".join(cells) + " |")
json.dump(out, open(D + f"recon_summary{TAG}.json", "w"), indent=1)
print(f"n_maps={n}\n| metric | model | " + " | ".join(f"P={p}" for p in P) + " |\n|---|---|" + "---|" * len(P))
print("\n".join(rows))
