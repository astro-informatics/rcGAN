"""Training curves of the new paper-mm runs from the CSV logger: epoch-mean generator / critic loss and validation PSNR
(P = 1 and 8 sample mean) per epoch. R and F100 have no local training logs (trained before CSV logging).
Writes figures/fig_mm_training.{png,pdf} and figures/training_curves.json.

Usage: python mm_loss_curves.py [MS MSS FNA MSNA]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CK = "/share/gpu0/jjwhit/crps_review/paper/mm/ckpt/"
FIG = "/share/gpu0/jjwhit/crps_review/paper/mm/figures/"
RUN = {"MS": "mmgan_crps_ms", "MSS": "mmgan_crps_ms_sigma", "FNA": "mmgan_crps_fair_noadv", "MSNA": "mmgan_crps_ms_noadv"}
COL = {"MS": "#1baf7a", "MSS": "#008300", "FNA": "#eda100", "MSNA": "#e87ba4"}
LAB = {"MS": "MS", "MSS": "MS-σ", "FNA": "F, no critic", "MSNA": "MS, no critic"}
LS = {"MS": "-", "MSS": "-", "FNA": "--", "MSNA": "--"}
models = sys.argv[1:] or list(RUN)

out = {}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for k in models:
    f = os.path.join(CK, RUN[k], "csv_logs/version_0/metrics.csv")
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    d["epoch"] = d["epoch"].ffill()  # validation rows carry no epoch; validation runs at the end of each training epoch
    tr = d.dropna(subset=["g_loss"]).groupby("epoch")[["g_loss", "d_loss"]].mean()
    va = d.dropna(subset=["psnr_8_step"]).groupby("epoch")[["psnr_8_step", "psnr_1_step"]].mean()
    out[k] = dict(epochs=int(tr.index.max()) + 1, g_loss=tr.g_loss.round(5).tolist(), d_loss=tr.d_loss.round(4).tolist(),
                  val_psnr8=va.psnr_8_step.round(3).tolist(), val_psnr1=va.psnr_1_step.round(3).tolist())
    kw = dict(color=COL[k], ls=LS[k], lw=2, label=LAB[k])
    ax[0].plot(tr.index, tr.g_loss, **kw)
    ax[1].plot(tr.index, tr.d_loss, **kw)
    ax[2].plot(va.index, va.psnr_8_step, **kw)
ax[0].set(xlabel="epoch", ylabel="generator loss (epoch mean)", yscale="symlog", title="generator: regulariser + adv. term")
ax[1].set(xlabel="epoch", ylabel="critic loss (epoch mean)", yscale="symlog", title="critic")
ax[2].set(xlabel="epoch", ylabel="val PSNR, 8-sample mean [dB]", title="validation")
for a in ax:
    a.grid(alpha=0.3, lw=0.5)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
ax[2].legend(frameon=False)
fig.text(0.01, 0.01, "Losses are different objectives per run (MS vs MS-σ vs fair CRPS): compare shapes, not levels.", fontsize=8, color="#555")
fig.tight_layout(rect=(0, 0.04, 1, 1))
os.makedirs(FIG, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(FIG + "fig_mm_training." + ext, dpi=200)
json.dump(out, open(FIG + "training_curves.json", "w"))
print({k: (v["epochs"], v["val_psnr8"][-1] if v["val_psnr8"] else None) for k, v in out.items()})
