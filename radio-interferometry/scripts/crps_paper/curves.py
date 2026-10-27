"""Training curves for the radio runs. Sources: Lightning CSV logs (R's CSV was truncated at ~epoch 38 when scancel killed
the logger mid-write, so the F backup copy is used for F), the end-of-epoch progress-bar values in the Slurm stdout
(g_loss, d_loss, psnr_8_step, psnr_1_step on the last validation batch), and beta (std_mult) stored in the kept checkpoints.

Usage: python curves.py R:<slurm.out> F:<slurm.out> MS:<slurm.out> ...
"""
import glob
import json
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = "/share/gpu0/jjwhit/crps_review/paper/radio/"
COLORS = {"R": "#1f77b4", "F": "#d62728", "MS": "#2ca02c", "Fnoadv": "#9467bd"}


def progress(path):
    """Last progress-bar update of every epoch: epoch -> dict(g_loss, d_loss, psnr_8_step, psnr_1_step, it/s)."""
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    out = {}
    pat = re.compile(r"^Epoch (\d+): 100%.*?\[(\S+)<\S+,\s+([\d.]+)it/s(.*)\]$")
    for line in txt.split("\n"):
        m = pat.match(line.strip())
        if m:
            e = int(m.group(1))
            kv = dict(re.findall(r"(\w+)=([-\d.e]+)", m.group(4)))
            out[e] = {k: float(v) for k, v in kv.items() if k in ("g_loss", "d_loss", "psnr_8_step", "psnr_1_step")}
            out[e]["its"] = float(m.group(3))
            out[e]["elapsed"] = m.group(2)
    return out


def main():
    runs = dict(a.split(":", 1) for a in sys.argv[1:])
    summ = {}
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    for t, out in runs.items():
        csvs = {"F": BASE + "curves/F_metrics_backup.csv"}
        csv = csvs.get(t, BASE + f"models/csv/radio_sim_{t}/version_0/metrics.csv")
        d = pd.read_csv(csv)
        v = d.dropna(subset=["val_psnr_8"]).groupby("epoch").last()
        r = d.dropna(subset=["reg_loss_epoch"]).groupby("epoch").last()
        pg = progress(out)
        beta = {}
        if t == "R":
            import torch
            for f in sorted(glob.glob(BASE + f"models/radio_sim_{t}/checkpoint-epoch=*.ckpt")):
                e = int(re.search(r"epoch=(\d+)", f).group(1))
                beta[e] = float(torch.load(f, map_location="cpu")["beta_std"])
        c = COLORS.get(t)
        ax[0].plot(v.index, v["val_psnr_8"], color=c, label=f"{t} val PSNR P=8")
        ax[0].plot(v.index, v["val_psnr_1"], color=c, ls="--", label=f"{t} val PSNR P=1")
        ax[1].plot(r.index, r["reg_loss_epoch"], color=c, label=t)
        ax[2].plot(r.index, r["adv_loss_epoch"], color=c, label=t)
        ep = sorted(pg)
        ax[3].plot(ep, [pg[e].get("d_loss", np.nan) for e in ep], color=c, label=t)
        if t == "R":
            ax[0].plot(ep, [pg[e].get("psnr_8_step", np.nan) for e in ep], color=c, ls=":", label="R last-batch PSNR P=8 (stdout)")
        summ[t] = {"csv_epochs": int(v.index.max()) + 1 if len(v) else 0,
                   "std_mult_csv": v["std_mult"].round(4).tolist(),
                   "std_mult_ckpt": beta,
                   "val_psnr_8_last": float(v["val_psnr_8"].iloc[-1]) if len(v) else None,
                   "epochs_stdout": len(pg), "mean_it_per_s": float(np.mean([pg[e]["its"] for e in ep])) if ep else None,
                   "d_loss_last": [pg[e].get("d_loss") for e in ep[-5:]], "g_loss_last": [pg[e].get("g_loss") for e in ep[-5:]],
                   "nan_or_inf_in_stdout": "nan" in open(out, errors="ignore").read().lower().replace("nanosec", "")}
    ax[0].set_title("validation PSNR")
    ax[1].set_title("regulariser (train, per epoch)")
    ax[1].set_yscale("log")
    ax[2].set_title("generator adversarial term (scheduled weight)")
    ax[2].set_yscale("symlog", linthresh=1e-4)
    ax[3].set_title("critic loss (end of epoch, stdout)")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(BASE + f"eval/fig_training_curves.{ext}", dpi=110)
    json.dump(summ, open(BASE + "eval/training_summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: {kk: vv for kk, vv in s.items() if kk != "std_mult_csv"} for k, s in summ.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
