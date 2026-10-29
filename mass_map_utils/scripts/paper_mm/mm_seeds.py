"""Seeds follow-up (BRIEF_mm_seeds.md): per-run metrics and mean +- sd over seeds, from the seeds-tagged analysis outputs
(job_analyse_seeds.sh: mm_analyse --tag seeds, a_analyse in eval/stageA_seeds, mm_recon_summary --tag seeds,
mm_paper_figs --out seeds) and mm/jointcov.json. Metric definitions are those of the main analysis; nothing is recomputed here.

Writes EVAL/seeds_summary.json, figures/seeds/fig_seeds_*.{png,pdf}, the key "seeds" in figures/paper_figs_data.json
(existing keys untouched), and prints markdown tables.
"""
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/share/gpu0/jjwhit/crps_review/paper/mm/"
EVAL, FIG = ROOT + "eval/", ROOT + "figures/seeds/"
os.makedirs(FIG, exist_ok=True)
# method -> runs (seed 0 first). R = original rcGAN run (older code), reported separately from the retrained R_s (paper branch).
GROUPS = {"F": ["F100", "F1", "F2"], "MS": ["MS", "MS1", "MS2"], "MS-σ": ["MSS", "MSS1", "MSS2"], "ES": ["ES0", "ES1", "ES2"],
          "R (paper branch)": ["R0", "R1", "R2"], "R (original)": ["R"], "F, no critic": ["FNA"], "MS, no critic": ["MSNA"]}
COL = {"F": "#eb6834", "MS": "#1baf7a", "MS-σ": "#008300", "ES": "#6250d6", "R (paper branch)": "#2a78d6", "R (original)": "#2a78d6",
       "F, no critic": "#eda100", "MS, no critic": "#e87ba4"}
JKEY = {"R": "R", "F100": "F", "MS": "MS", "MSS": "MSsig", "FNA": "Fnoadv", "MSNA": "MSnoadv", "ES0": "ES"}
JKEY.update({f"{b}{s}": f"{j}_s{s}" for b, j in [("ES", "ES"), ("F", "F"), ("MS", "MS"), ("MSS", "MSsig")] for s in (1, 2)})
JKEY.update({f"R{s}": f"R_s{s}" for s in range(3)})
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e6e3"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "lines.linewidth": 1.6, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "figure.facecolor": "white"})


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else {}


J = jload(EVAL + "joint/analysis_seeds/joint.json")
SA = jload(EVAL + "stageA_seeds/analysis/stageA.json")
RC = jload(EVAL + "recon/recon_summary_seeds.json")
PF = jload(ROOT + "figures/seeds/paper_figs_data.json")
JC = jload(ROOT + "jointcov.json").get("methods", {})
ELLS = ["l300", "l4869", "l19424", "l27432"]
PATCH = [1, 2, 4, 8, 16, 32, 64, "footprint"]


def best_epoch(k):
    """best val RMSE epoch from the run's validate log (val_<K>_*.out or the main-run val logs)."""
    import glob
    for f in sorted(glob.glob(ROOT + f"logs/val_{k}_*.out")):
        for line in open(f, errors="ignore"):
            if line.startswith("BEST EPOCH FOR RMSE:"):
                return int(line.split(":")[1])
    return None


def run_metrics(k):
    m = {}
    if k in RC.get("psnr", {}):
        P = RC["P"]
        for met in ["psnr", "rmse", "ssim"]:
            for j, p in enumerate(P):
                m[f"{met}_P{p}"] = RC[met][k]["val"][j]
    if k in J.get("scores", {}).get("es_full", {}):
        m["energy_score"] = J["scores"]["es_full"][k]["val"]
        m["vs_local"] = J["scores"]["vs_local"][k]["val"]
        m["err2_over_var"] = J["spread_skill"]["pixel_err2_over_var"][k]["val"]
        hs = J["spectra"]["per_sample_highell_ratio"].get(k)
        if hs:
            m["per_sample_power_highell_median"] = hs["median"]
        for lab in ELLS:
            m[f"power_ratio_{lab}"] = J["spectra"]["mean_sample_over_truth_mask"][lab][k]["val"]
        m["rho_median"] = J["diversity"]["median_rho"].get(k)
    if k in SA:
        m["pixel_abs_dcov_pp"] = 100 * float(np.mean([abs(e["cov"] - e["exact"]) for e in SA[k]["A1_cov"]]))
        m["pixel_cov90_stageA_pp"] = 100 * next(e["cov"] - e["exact"] for e in SA[k]["A1_cov"] if abs(e["level"] - 0.9) < 1e-9)
    if k in PF.get("patch", {}):
        pt = PF["patch"][k]
        for p, v in zip(pt["patch"], pt["cov90_minus_exact"]):
            m[f"patch_cov90_{p}"] = 100 * v
    jk = JKEY.get(k)
    if jk in JC:
        lv = np.array(JC[jk]["model"]["levels"])
        i = int(np.argmin(np.abs(lv - 0.9)))
        for st in ["maxz", "depth", "avgrank"]:
            for s in ["64", "full"]:
                e = JC[jk]["model"]["stats"][st].get(s)
                if e:
                    m[f"jc_{st}_{s}"] = 100 * (e["cov"][i] - 0.9)
    m["best_epoch"] = best_epoch(k)
    return m


runs = {k: run_metrics(k) for g in GROUPS.values() for k in g}
agg = {}
for g, ks in GROUPS.items():
    have = [k for k in ks if len(runs[k]) > 1]
    agg[g] = {"runs": have, "mean": {}, "sd": {}}
    for met in sorted({mm for k in have for mm in runs[k]}):
        v = np.array([runs[k][met] for k in have if runs[k].get(met) is not None], float)
        if len(v):
            agg[g]["mean"][met] = float(v.mean())
            agg[g]["sd"][met] = float(v.std(ddof=1)) if len(v) > 1 else None
            agg[g].setdefault("n", {})[met] = len(v)
out = {"groups": GROUPS, "runs": runs, "summary": agg,
       "notes": "sd = sample sd over seeds (ddof 1). Patch cov90: 90% patch-mean coverage - exact (pp), 300 joint maps; pixel |dcov|: "
                "stage-A mean |coverage - exact| over levels 5..95% (pp, 1000 maps); energy score: fair ES, 300 maps, N = 256; "
                "power ratio: <P_sample>/P_truth (masked) at the given ell; jc_*: jointcov 90% coverage - 90 (pp)."}
json.dump(out, open(EVAL + "seeds_summary.json", "w"), indent=1)

# paper_figs_data.json: add/replace only the "seeds" key
pfd = ROOT + "figures/paper_figs_data.json"
d = jload(pfd)
d["seeds"] = out
json.dump(d, open(pfd, "w"), indent=1)

# ---------------------------------------------------------------- tables
COLS = [("psnr_P32", "PSNR P=32", 2), ("rmse_P32", "RMSE P=32 (×1e3)", 3), ("energy_score", "energy score", 3),
        ("pixel_abs_dcov_pp", "pixel |Δcov| pp", 2), ("patch_cov90_16", "16 px", 1), ("patch_cov90_64", "64 px", 1),
        ("patch_cov90_footprint", "footprint", 1), ("power_ratio_l4869", "P ratio ℓ4.9k", 2), ("power_ratio_l27432", "P ratio ℓ27k", 2),
        ("jc_maxz_64", "maxz 64", 1), ("jc_depth_full", "depth fp", 1), ("best_epoch", "best ep", 0)]


def cell(v, nd, met):
    if v is None:
        return "–"
    if met.startswith("rmse"):
        v = 1e3 * v
    return f"{v:+.{nd}f}" if met.startswith(("patch", "jc_")) else f"{v:.{nd}f}"


L = ["| run | " + " | ".join(c[1] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
for g, ks in GROUPS.items():
    for k in ks:
        if len(runs[k]) > 1:
            L.append(f"| {k} | " + " | ".join(cell(runs[k].get(m), nd, m) for m, _, nd in COLS) + " |")
L += ["", "| method (n seeds) | " + " | ".join(c[1] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
for g, a in agg.items():
    if not a["runs"]:
        continue
    cs = []
    for m, _, nd in COLS:
        mu, sd = a["mean"].get(m), a["sd"].get(m)
        sc = 1e3 if m.startswith("rmse") else 1
        cs.append("–" if mu is None else (f"{sc * mu:.{nd}f}" + (f" ± {sc * sd:.{nd}f}" if sd is not None else "")))
    L.append(f"| {g} ({len(a['runs'])}) | " + " | ".join(cs) + " |")
open(EVAL + "seeds_tables.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ---------------------------------------------------------------- figures
# (1) 90% patch-mean coverage error vs patch size: each seed thin, method mean thick
fig, ax = plt.subplots(figsize=(5.2, 3.4))
x = np.arange(len(PATCH))
for g, ks in GROUPS.items():
    ys = [[runs[k].get(f"patch_cov90_{p}") for p in PATCH] for k in ks if f"patch_cov90_64" in runs[k]]
    if not ys:
        continue
    ls = "--" if "no critic" in g else ("-." if g == "R (original)" else "-")
    for y in ys:
        ax.plot(x, y, color=COL[g], lw=0.7, alpha=0.5, ls=ls)
    ax.plot(x, np.mean(ys, 0), color=COL[g], lw=1.8, ls=ls, marker="o", ms=3, label=f"{g} ({len(ys)})")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([str(p) if p != "footprint" else "fp" for p in PATCH])
ax.set_xlabel("patch side (px; fp = survey footprint)")
ax.set_ylabel("90% coverage − nominal (pp)")
ax.set_title("patch-mean coverage over seeds (thin: seeds, thick: mean)")
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
for ext in ("png", "pdf"):
    fig.savefig(FIG + f"fig_seeds_patch_cov.{ext}", dpi=200)
plt.close(fig)

# (2) per-metric strip plot: seeds as dots, method mean as a bar
SM = [("psnr_P32", "PSNR P=32 (dB)"), ("energy_score", "energy score"), ("pixel_abs_dcov_pp", "pixel |Δcov| (pp)"),
      ("patch_cov90_64", "64 px cov90 − 90 (pp)"), ("patch_cov90_footprint", "footprint cov90 − 90 (pp)"),
      ("power_ratio_l27432", "power ratio ℓ ≈ 27k")]
gl = [g for g in GROUPS if agg[g]["runs"]]
fig, axs = plt.subplots(1, len(SM), figsize=(2.0 * len(SM), 3.2))
for ax, (m, t) in zip(axs, SM):
    for i, g in enumerate(gl):
        v = [runs[k][m] for k in agg[g]["runs"] if runs[k].get(m) is not None]
        if v:
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=14, color=COL[g], zorder=3)
            ax.hlines(np.mean(v), i - 0.3, i + 0.3, color=COL[g], lw=1.8)
    ax.set_xticks(range(len(gl)))
    ax.set_xticklabels(gl, rotation=60, ha="right", fontsize=6.5)
    ax.set_title(t)
    ax.grid(axis="x", visible=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG + f"fig_seeds_metrics.{ext}", dpi=200)
plt.close(fig)
print("wrote", EVAL + "seeds_summary.json", FIG)
