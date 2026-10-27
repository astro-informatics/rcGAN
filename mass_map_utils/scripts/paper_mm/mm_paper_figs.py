"""Paper figures for the mass-mapping experiments (PNG + PDF), from the joint-test outputs
(EVAL/joint/main/<key>/, 300 maps x N = 256, paired) and, where available, the stage-A
per-pixel ranks (EVAL/stageA/stats/<key>/pix_rank.npy, 1000 maps).

  fig_mm_coverage    coverage - exact level vs level: per pixel, 16-px and 64-px patch means
  fig_mm_power       per-sample power / truth and posterior-mean transfer vs ell
  fig_mm_patch_cov   90% coverage - exact and mean z^2 vs patch size
  fig_mm_gallery     truth, posterior mean, 3 samples, posterior SD for the same map

Bootstrap: B = 2000 resamples over maps, rng seed 0, same indices for every model.
Usage: python mm_paper_figs.py [--models R F100 MS FNA MSNA] [--map 0]
"""
import argparse
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVAL = "/share/gpu0/jjwhit/crps_review/paper/mm/eval/"
FIG = "/share/gpu0/jjwhit/crps_review/paper/mm/figures/"
p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["R", "F100", "MS", "FNA", "MSNA"])
p.add_argument("--map", type=int, default=0)
p.add_argument("--B", type=int, default=2000)
args = p.parse_args()
MOD = [k for k in args.models if os.path.exists(os.path.join(EVAL, "joint/main", k, "meta.json"))]
# categorical slots in fixed order (blue, orange, aqua, yellow, magenta); colour follows the method
COL = {"R": "#2a78d6", "F100": "#eb6834", "MS": "#1baf7a", "FNA": "#eda100", "MSNA": "#e87ba4"}
LAB = {"R": "rcGAN (R)", "F100": "fair CRPS (F)", "MS": "multiscale fair CRPS (MS)", "FNA": "F, no critic", "MSNA": "MS, no critic"}
LS = {"R": "-", "F100": "-", "MS": "-", "FNA": "--", "MSNA": "--"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e6e3"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "lines.linewidth": 1.6, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "figure.facecolor": "white"})
os.makedirs(FIG, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", name)


def load(k, name, sub="joint/main"):
    return np.load(os.path.join(EVAL, sub, k, name + ".npy"), mmap_mode="r")


meta = json.load(open(os.path.join(EVAL, "joint/main", MOD[0], "meta.json")))
N, n_maps = meta["n_samp"], min(json.load(open(os.path.join(EVAL, "joint/main", k, "meta.json")))["n_done"] for k in MOD)
rng = np.random.default_rng(0)
C = np.stack([np.bincount(rng.integers(0, n_maps, n_maps), minlength=n_maps) for _ in range(args.B)]).astype(float)


def boot_ratio(num, den, Cm=None):
    """num, den: (n, ...) per-map sums -> value, (lo, hi) of sum(num)/sum(den)."""
    Cm = C if Cm is None else Cm
    v = num.sum(0) / den.sum(0)
    d = (Cm @ num.reshape(len(num), -1)) / (Cm @ den.reshape(len(den), -1))
    return v, np.percentile(d, [2.5, 97.5], 0).reshape((2,) + v.shape)


# ------------------------------------------------------------------ coverage vs level
LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)
kq = np.array([max(1, int(round((N + 1) * (1 - L) / 2))) for L in LEVELS])
exact = (N + 1 - 2 * kq) / (N + 1)


def cov_curve(r):
    """r: (n, T) ranks of truth among N samples -> per-map covered counts (n, levels) and counts (n,)."""
    r = np.asarray(r)
    hit = np.stack([((r >= k) & (r <= N - k)).sum(1) for k in kq], 1).astype(float)
    return hit, np.full((len(r), 1), r.shape[1], dtype=float)


stageA = all(os.path.exists(os.path.join(EVAL, "stageA/stats", k, "pix_rank.npy")) for k in MOD)
panels = [("pix", "per pixel" + (" (1000 maps)" if stageA else f" ({n_maps} maps)")), ("pm16", "16-px patch mean (4.6′)"),
          ("pm64", "64-px patch mean (18.6′)")]
covres = {}
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True)
for ax, (nm, tt) in zip(axs, panels):
    for k in MOD:
        if nm == "pix" and stageA:
            r = load(k, "pix_rank", "stageA/stats")
            nA = r.shape[0]
            rngA = np.random.default_rng(0)
            CA = np.stack([np.bincount(rngA.integers(0, nA, nA), minlength=nA) for _ in range(args.B)]).astype(float)
            hit, cnt = cov_curve(np.asarray(r))
            v, (lo, hi) = boot_ratio(hit, np.repeat(cnt, len(LEVELS), 1), CA)
        else:
            key = "pm1_rank" if nm == "pix" else f"{nm}_rank"
            hit, cnt = cov_curve(load(k, key)[:n_maps])
            v, (lo, hi) = boot_ratio(hit, np.repeat(cnt, len(LEVELS), 1))
        covres.setdefault(nm, {})[k] = {"levels": exact.tolist(), "cov_minus_exact": (v - exact).tolist(),
                                        "lo": (lo - exact).tolist(), "hi": (hi - exact).tolist()}
        ax.plot(exact, 100 * (v - exact), color=COL[k], ls=LS[k], label=LAB[k])
        ax.fill_between(exact, 100 * (lo - exact), 100 * (hi - exact), color=COL[k], alpha=0.18, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(tt)
    ax.set_xlabel("nominal level")
axs[0].set_ylabel("coverage − level (pp)")
axs[0].legend(frameon=False, loc="best")
fig.tight_layout()
save(fig, "fig_mm_coverage")

# ------------------------------------------------------------------ power spectra
k_eff = np.array(meta["k_eff"])
ell = k_eff * 2 * np.pi / np.deg2rad(300 * 0.29 / 60)
ok = ell < 2.7e4  # drop pixel-scale bins (wavelength < 3.3 px)
specres = {}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
for k in MOD:
    bs, bx = np.asarray(load(k, "bp_mask_s")[:n_maps]).mean(1), np.asarray(load(k, "bp_mask_x")[:n_maps])
    v, (lo, hi) = boot_ratio(bs, bx)
    cx = np.asarray(load(k, "bp_mask_cross_mx")[:n_maps])
    t, (tlo, thi) = boot_ratio(cx, bx)
    specres[k] = {"ell": ell.tolist(), "sample_over_truth": v.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                  "transfer": t.tolist(), "transfer_lo": tlo.tolist(), "transfer_hi": thi.tolist()}
    for ax, (a, l_, h_) in zip(axs, [(v, lo, hi), (t, tlo, thi)]):
        ax.plot(ell[ok], a[ok], color=COL[k], ls=LS[k], marker="o", ms=2.5, label=LAB[k])
        ax.fill_between(ell[ok], l_[ok], h_[ok], color=COL[k], alpha=0.18, lw=0)
for ax, tt in zip(axs, ["mean per-sample power / truth power", "posterior-mean transfer  Σ P(m,x) / Σ P(x)"]):
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("multipole ℓ")
    ax.set_title(tt)
axs[0].legend(frameon=False)
fig.tight_layout()
save(fig, "fig_mm_power")

# ------------------------------------------------------------------ patch-size coverage
PS = [1, 2, 4, 8, 16, 32, 64]
k90 = max(1, int(round((N + 1) * 0.05)))
ex90 = (N + 1 - 2 * k90) / (N + 1)
patchres = {}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
xs = PS + [300]
for k in MOD:
    cov, z2 = [], []
    for s in PS + ["global"]:
        if s == "global":
            gs, gx = np.asarray(load(k, "gm_s")[:n_maps]), np.asarray(load(k, "gm_x")[:n_maps])
            r = (gs < gx[:, None]).sum(1)[:, None]
            z = ((gx - gs.mean(1)) / gs.std(1, ddof=1))[:, None]
        else:
            r, z = np.asarray(load(k, f"pm{s}_rank")[:n_maps]), np.asarray(load(k, f"pm{s}_z")[:n_maps], dtype=float)
        cnt = np.full((n_maps, 1), r.shape[1], dtype=float)
        hit = ((r >= k90) & (r <= N - k90)).sum(1, keepdims=True).astype(float)
        cov.append(boot_ratio(hit, cnt))
        z2.append(boot_ratio((z ** 2).sum(1, keepdims=True) / (1 + 1 / N), cnt))
    cv = np.array([c[0][0] for c in cov]) - ex90
    clo, chi = np.array([c[1][0][0] for c in cov]) - ex90, np.array([c[1][1][0] for c in cov]) - ex90
    zv, zlo, zhi = np.array([c[0][0] for c in z2]), np.array([c[1][0][0] for c in z2]), np.array([c[1][1][0] for c in z2])
    patchres[k] = {"patch": PS + ["footprint"], "cov90_minus_exact": cv.tolist(), "lo": clo.tolist(), "hi": chi.tolist(),
                   "mean_z2": zv.tolist(), "z2_lo": zlo.tolist(), "z2_hi": zhi.tolist()}
    axs[0].plot(xs, 100 * cv, color=COL[k], ls=LS[k], marker="o", ms=3, label=LAB[k])
    axs[0].fill_between(xs, 100 * clo, 100 * chi, color=COL[k], alpha=0.18, lw=0)
    axs[1].plot(xs, zv, color=COL[k], ls=LS[k], marker="o", ms=3, label=LAB[k])
    axs[1].fill_between(xs, zlo, zhi, color=COL[k], alpha=0.18, lw=0)
for ax, tt in zip(axs, ["90% interval: coverage − level (pp)", "mean z² of patch mean (1 = calibrated)"]):
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in PS] + ["all"])
    ax.minorticks_off()
    ax.set_xlabel("patch size (px; 1 px = 0.29′; all = footprint mean)")
    ax.set_title(tt)
axs[0].axhline(0, color=MUTED, lw=0.8)
axs[1].axhline(1, color=MUTED, lw=0.8)
axs[1].set_yscale("log")
axs[0].legend(frameon=False)
fig.tight_layout()
save(fig, "fig_mm_patch_cov")

# ------------------------------------------------------------------ gallery
i = args.map
gal = [k for k in MOD if os.path.exists(os.path.join(EVAL, "joint/main", k, "fig_samples.npz"))]
Z = {k: np.load(os.path.join(EVAL, "joint/main", k, "fig_samples.npz")) for k in gal}
msk = Z[gal[0]]["mask"].astype(bool)
x = Z[gal[0]][f"x{i}"]
for k in gal:
    assert np.allclose(Z[k][f"x{i}"], x), "gallery truth differs between models (not paired)"
vlim = np.percentile(x[msk], [1, 99.5])
smax = max(np.percentile(Z[k][f"s{i}"][msk], 99.5) for k in gal)
cols = ["truth", "posterior mean", "sample 1", "sample 2", "sample 3", "posterior SD"]
fig, axs = plt.subplots(len(gal), 6, figsize=(7.2, 1.3 * len(gal) + 0.35), squeeze=False)
for r_, k in enumerate(gal):
    z = Z[k]
    ims = [x, z[f"m{i}"]] + list(z[f"samp{i}"][:3]) + [z[f"s{i}"]]
    for c_, im in enumerate(ims):
        ax = axs[r_, c_]
        sd = c_ == 5
        h = ax.imshow(np.where(msk, im, np.nan), cmap="magma" if sd else "viridis", vmin=0 if sd else vlim[0],
                      vmax=smax if sd else vlim[1], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        if r_ == 0:
            ax.set_title(cols[c_])
    axs[r_, 0].set_ylabel(LAB[k], fontsize=7)
fig.subplots_adjust(wspace=0.03, hspace=0.06, right=0.9)
cax1 = fig.add_axes([0.91, 0.55, 0.012, 0.3])
cax2 = fig.add_axes([0.91, 0.12, 0.012, 0.3])
fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(*vlim), cmap="viridis"), cax=cax1, label="κ")
fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, smax), cmap="magma"), cax=cax2, label="SD(κ)")
save(fig, "fig_mm_gallery")

json.dump({"models": MOD, "n_maps": n_maps, "N": N, "pixel_coverage_source": "stageA 1000 maps" if stageA else "joint pm1",
           "coverage": covres, "power": specres, "patch": patchres}, open(os.path.join(FIG, "paper_figs_data.json"), "w"), indent=1)
