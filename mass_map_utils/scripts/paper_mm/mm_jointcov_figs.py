"""Per-domain joint-coverage figures for WL (BRIEF_jointcov.md), from mm/jointcov.json -> mm/figures/jointcov/ (PNG + PDF).

  fig_jc_marginal   per-pixel coverage - nominal vs nominal, every variant (null check in grey)
  fig_jc_full       whole-footprint coverage - nominal vs nominal for maxz / depth / avgrank
  fig_jc_vs_size    90% coverage - 90 vs region size for mean / maxz / depth / avgrank
Grey band: +-2 binomial SE for the number of regions (pixels are correlated, so it is a lower bound on the MC error).
"""
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/share/gpu0/jjwhit/crps_review/paper/mm/"
FIG = ROOT + "figures/jointcov/"
os.makedirs(FIG, exist_ok=True)
D = json.load(open(ROOT + "jointcov.json"))
M = D["methods"]
# categorical slots in fixed order, colour follows the method (as fig_mm_*); ES = slot 7 (violet)
COL = {"R": "#2a78d6", "F": "#eb6834", "MS": "#1baf7a", "Fnoadv": "#eda100", "MSnoadv": "#e87ba4", "MSsig": "#008300",
       "ES": "#6250d6"}
ORDER = ["R", "F", "MS", "MSsig", "ES", "Fnoadv", "MSnoadv"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e6e3"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "lines.linewidth": 1.6, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "figure.facecolor": "white"})


def base(j):
    return j.split("_s")[0]


def style(j):
    b, seeded = base(j), "_s" in j
    ls = "--" if b.endswith("noadv") else ("-." if b == "R" and seeded else "-")
    return dict(color=COL[b], ls=ls, lw=0.9 if seeded else 1.6, alpha=0.75 if seeded else 1.0)


keys = sorted(M, key=lambda j: (ORDER.index(base(j)), j))
lv = np.array(next(iter(M.values()))["model"]["levels"])
i90 = int(np.argmin(np.abs(lv - 0.9)))


def curve(j, st, k, role="model"):
    e = M[j][role]["stats"][st][str(k)]
    return np.array(e["cov"]) - lv, e["n"]


def band(ax, n):
    se = 2 * np.sqrt(lv * (1 - lv) / n)
    ax.fill_between(lv, -100 * se, 100 * se, color=GRID, lw=0, zorder=0)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG + f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", name)


def finish(ax, title):
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("nominal level")


# (i) marginal
fig, ax = plt.subplots(figsize=(4.2, 3.0))
band(ax, curve(keys[0], "pixel", 1)[1])
ax.plot(lv, 100 * curve(keys[0], "pixel", 1, "null")[0], color=MUTED, ls=":", lw=1.2, label="null check")
for j in keys:
    ax.plot(lv, 100 * curve(j, "pixel", 1)[0], label=D["labels"][j], **style(j))
finish(ax, "per-pixel coverage (marginal)")
ax.set_ylabel("empirical − nominal (pp)")
ax.legend(frameon=False, ncol=1, loc="best")
save(fig, "fig_jc_marginal")

# (ii) whole footprint
fig, axs = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
for ax, st, t in zip(axs, ["maxz", "depth", "avgrank"],
                     ["max |z| band\n(simultaneous)", "depth\n(energy-score geometry)", "average rank\n(dependence)"]):
    band(ax, curve(keys[0], st, "full")[1])
    ax.plot(lv, 100 * curve(keys[0], st, "full", "null")[0], color=MUTED, ls=":", lw=1.2, label="null check")
    for j in keys:
        ax.plot(lv, 100 * curve(j, st, "full")[0], label=D["labels"][j], **style(j))
    finish(ax, f"footprint: {t}")
axs[0].set_ylabel("empirical − nominal (pp)")
axs[-1].legend(frameon=False, loc="best")
save(fig, "fig_jc_full")

# (iii) 90% error vs region size
SIZES = [1, 2, 4, 8, 16, 32, 64, 128, "full"]
fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.0), sharey=True)
for ax, st in zip(axs, ["mean", "maxz", "depth", "avgrank"]):
    for role, kw in [("null", dict(color=MUTED, ls=":", lw=1.2)), ("model", None)]:
        for j in (keys if role == "model" else keys[:1]):
            st_ = M[j][role]["stats"][st]
            xs = [i for i, s in enumerate(SIZES) if str(s) in st_]
            ys = [100 * (st_[str(SIZES[i])]["cov"][i90] - 0.9) for i in xs]
            ax.plot(xs, ys, marker="o", ms=3, **(kw or style(j)), label="null check" if role == "null" else D["labels"][j])
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([str(s) if s != "full" else "all" for s in SIZES])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(f"{st}: 90% coverage − 90")
    ax.set_xlabel("region side (px; all = footprint)")
axs[0].set_ylabel("pp")
axs[-1].legend(frameon=False, loc="lower left", fontsize=6.5)
save(fig, "fig_jc_vs_size")
