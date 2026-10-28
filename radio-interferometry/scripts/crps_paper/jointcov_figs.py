"""Per-domain jointcov figures for radio (BRIEF_jointcov): one row per variant file (raw full field, positivity-clipped, source mask).
(i)   fig_marginal:  per-pixel coverage - nominal vs nominal, all methods (null check dashed grey)
(ii)  fig_fullimage: full-image coverage - nominal for maxz / depth / avgrank (and mean), +-2 binomial MC sd shaded
(iii) fig_cov90_vs_scale: 90% coverage - 0.9 vs region size k for every statistic

Usage: python jointcov_figs.py [base_dir]   (default /share/gpu0/jjwhit/crps_review/paper/radio/)
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "/share/gpu0/jjwhit/crps_review/paper/radio/"
OUT = BASE + "figures/jointcov/"
FILES = [("jointcov.json", "raw samples, full field"), ("jointcov_clip.json", "positivity-clipped, full field"),
         ("jointcov_srcmask.json", "raw samples, source mask")]
COLORS = {"R": "#1f77b4", "F": "#d62728", "MS": "#2ca02c", "MSsig": "#ff7f0e", "Fnoadv": "#9467bd"}
STATS = ["pixel", "mean", "maxz", "depth", "avgrank"]
KORDER = ["1", "2", "4", "8", "16", "32", "64", "full"]


def load():
    out = []
    for f, ttl in FILES:
        if os.path.exists(BASE + f):
            out.append((json.load(open(BASE + f)), ttl))
    return out


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT + f"{name}.{ext}", dpi=110, bbox_inches="tight")
    plt.close(fig)


def curve(res, part, st, k):
    s = (res.get(part) or {}).get("stats", {}).get(st, {}).get(k)
    return None if s is None else (np.array(s["cov"]), s["n"])


def main():
    D = load()
    if not D:
        sys.exit("no jointcov json")
    # (i) marginal
    fig, ax = plt.subplots(1, len(D), figsize=(5 * len(D), 4), squeeze=False)
    for a_, (d, ttl) in zip(ax[0], D):
        for m, res in d["methods"].items():
            lv = np.array(res["model"]["levels"])
            c = curve(res, "model", "pixel", "1")
            a_.plot(lv, c[0] - lv, "o-", ms=3, color=COLORS.get(m), label=d["labels"].get(m, m))
            n = curve(res, "null", "pixel", "1")
            if n is not None:
                a_.plot(lv, n[0] - lv, "--", lw=0.8, color=COLORS.get(m), alpha=0.6)
        a_.axhline(0, color="k", lw=0.8)
        a_.set_title(f"per-pixel coverage: {ttl}", fontsize=10)
        a_.set_xlabel("nominal level")
        a_.set_ylabel("empirical - nominal")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("radio: marginal coverage (dashed: null check, truth replaced by a sample)", fontsize=10)
    save(fig, "fig_marginal")
    # (ii) full image
    fst = ["mean", "maxz", "depth", "avgrank"]
    fig, ax = plt.subplots(len(D), len(fst), figsize=(4.2 * len(fst), 3.6 * len(D)), squeeze=False)
    for r, (d, ttl) in enumerate(D):
        n_cases = None
        for j, st in enumerate(fst):
            a_ = ax[r, j]
            for m, res in d["methods"].items():
                lv = np.array(res["model"]["levels"])
                c = curve(res, "model", st, "full")
                if c is None:
                    continue
                n_cases = c[1]
                a_.plot(lv, c[0] - lv, "o-", ms=3, color=COLORS.get(m), label=d["labels"].get(m, m))
                nl = curve(res, "null", st, "full")
                if nl is not None:
                    a_.plot(lv, nl[0] - lv, "--", lw=0.8, color=COLORS.get(m), alpha=0.6)
            if n_cases:
                sd = np.sqrt(lv * (1 - lv) / n_cases)
                a_.fill_between(lv, -2 * sd, 2 * sd, color="grey", alpha=0.25, lw=0)
            a_.axhline(0, color="k", lw=0.8)
            side = d["methods"][next(iter(d["methods"]))]["model"]["sides"][st]
            a_.set_title(f"{st} ({side}), full image\n{ttl}", fontsize=9)
            a_.set_xlabel("nominal level")
            if j == 0:
                a_.set_ylabel("empirical - nominal")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig_fullimage")
    # (iii) 90% error vs scale
    fig, ax = plt.subplots(len(D), len(STATS), figsize=(4 * len(STATS), 3.4 * len(D)), squeeze=False)
    for r, (d, ttl) in enumerate(D):
        for j, st in enumerate(STATS):
            a_ = ax[r, j]
            for m, res in d["methods"].items():
                lv = list(res["model"]["levels"])
                i90 = lv.index(0.9)
                ks = [k for k in KORDER if curve(res, "model", st, k) is not None]
                xs = [KORDER.index(k) for k in ks]
                a_.plot(xs, [curve(res, "model", st, k)[0][i90] - 0.9 for k in ks], "o-", ms=3, color=COLORS.get(m),
                        label=d["labels"].get(m, m))
                a_.plot(xs, [curve(res, "null", st, k)[0][i90] - 0.9 for k in ks], "--", lw=0.8, color=COLORS.get(m), alpha=0.5)
            a_.axhline(0, color="k", lw=0.8)
            a_.set_xticks(range(len(KORDER)))
            a_.set_xticklabels(KORDER, fontsize=7)
            a_.set_ylim(-0.95, 0.15)
            a_.set_title(f"{st}: 90% coverage error\n{ttl}", fontsize=9)
            a_.set_xlabel("region size k [px]")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig_cov90_vs_scale")
    print("figures in", OUT)


if __name__ == "__main__":
    main()
