"""paper-mm copy of ~/crps_review/joint_tests/jt_analyse.py (paths, colours, model list, and a paired
difference vs F100 added; statistics unchanged).

CPU analysis for the joint tests. Reads OUT/main/<key>/, OUT/epochs/, OUT/latent/;
writes OUT/analysis/joint.json, OUT/analysis/summary.txt and figures to OUT/figures/.

Every number is a statistic g(sum over maps of a per-map vector). Bootstrap = resample
maps with replacement (B = 2000, seed 0) with the SAME resampled indices for every model,
so paired differences (model - R) get CIs directly.

Usage: python jt_analyse.py [--models R F100 S FB NL] [--sub main]
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats as sst

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "/share/gpu0/jjwhit/crps_review/paper/mm/eval/joint/"
p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["R", "F100", "MS", "MSS", "FNA", "MSNA"])
p.add_argument("--sub", default="main")
p.add_argument("--B", type=int, default=2000)
p.add_argument("--tag", default="", help="output suffix (analysis_<tag>, figures_<tag>), e.g. seeds; inputs still from --sub")
args = p.parse_args()
MOD = args.models
COL = {"R": "#2a78d6", "F100": "#eb6834", "MS": "#1baf7a", "MSS": "#008300", "FNA": "#eda100", "MSNA": "#e87ba4", "S": "#8c8c8c", "FB": "#555555", "NL": "#b0b0b0"}
COL.update({"F": COL["F100"], "ES": "#6250d6"})
for k in MOD:  # seed runs (ES0, F1, R2, MSS1, ...) take their method's colour
    COL.setdefault(k, COL[k.rstrip("0123456789")])
tag = args.tag or ("" if args.sub == "main" else args.sub)
FIG = os.path.join(OUT, f"figures_{tag}" if tag else "figures")
AN = os.path.join(OUT, f"analysis_{tag}" if tag else "analysis")
os.makedirs(FIG, exist_ok=True)
os.makedirs(AN, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "lines.linewidth": 2})


def load(key):
    d = os.path.join(OUT, args.sub, key)
    meta = json.load(open(os.path.join(d, "meta.json")))
    arr = {os.path.basename(f)[:-4]: np.load(f, mmap_mode="r") for f in glob.glob(os.path.join(d, "*.npy"))}
    return meta, arr


data = {k: load(k) for k in MOD}
n_maps = min(m["n_done"] for m, _ in data.values())
N = data[MOD[0]][0]["n_samp"]
k_eff = np.array(data[MOD[0]][0]["k_eff"])
ELL_PER_K = 2 * np.pi / np.deg2rad(300 * 0.29 / 60)
ell = k_eff * ELL_PER_K
nb = len(k_eff)
rng = np.random.default_rng(0)
C = np.stack([np.bincount(rng.integers(0, n_maps, n_maps), minlength=n_maps) for _ in range(args.B)]).astype(float)  # (B, n)
RES = {"n_maps": n_maps, "n_samp": N, "B": args.B, "ell_eff": ell.tolist()}


def boot(a, g):
    """a: (n, ...) per-map aggregates; g: function of the summed aggregate -> scalar/array.
    Returns (value, bootstrap draws)."""
    a = np.asarray(a, dtype=float)[:n_maps]
    flat = a.reshape(n_maps, -1)
    tot = flat.sum(0).reshape(a.shape[1:])
    draws = (C @ flat).reshape((args.B,) + a.shape[1:])
    return np.asarray(g(tot)), np.array([g(d) for d in draws])


def ci(draws):
    return np.percentile(draws, [2.5, 97.5], axis=0)


def summ(key_vals):
    """{model: (val, draws)} -> json-ready dict with CI and paired diff vs R and F100-R."""
    out = {}
    for k, (v, d) in key_vals.items():
        lo, hi = ci(d)
        e = {"val": np.round(v, 6).tolist(), "lo": np.round(lo, 6).tolist(), "hi": np.round(hi, 6).tolist()}
        if k != "R" and "R" in key_vals:
            dd = d - key_vals["R"][1]
            dlo, dhi = ci(dd)
            e["minus_R"] = np.round(v - key_vals["R"][0], 6).tolist()
            e["minus_R_lo"], e["minus_R_hi"] = np.round(dlo, 6).tolist(), np.round(dhi, 6).tolist()
        if k not in ("R", "F100") and "F100" in key_vals:
            dd = d - key_vals["F100"][1]
            dlo, dhi = ci(dd)
            e["minus_F100"] = np.round(v - key_vals["F100"][0], 6).tolist()
            e["minus_F100_lo"], e["minus_F100_hi"] = np.round(dlo, 6).tolist(), np.round(dhi, 6).tolist()
        out[k] = e
    return out


def fmt(e, nd=3, diff=False):
    if diff:
        return f"{e['minus_R']:+.{nd}f} [{e['minus_R_lo']:+.{nd}f}, {e['minus_R_hi']:+.{nd}f}]"
    return f"{e['val']:.{nd}f} [{e['lo']:.{nd}f}, {e['hi']:.{nd}f}]"


# ---------------------------------------------------------------- rank calibration machinery
NBIN = 10
rvals = np.arange(N + 1)
rbin = np.minimum((rvals * NBIN) // (N + 1), NBIN - 1)
p_bin = np.bincount(rbin, minlength=NBIN) / (N + 1)
k50, k90 = [max(1, int(round((N + 1) * (1 - L) / 2))) for L in (0.5, 0.9)]
exact50, exact90 = (N + 1 - 2 * k50) / (N + 1), (N + 1 - 2 * k90) / (N + 1)
# per-map aggregate vector: [hist(10), sum u, sum |u-.5|, cov50, cov90, count]


def rank_agg(r):
    """r: (n, T) integer ranks -> (n, NBIN + 5) per-map aggregates."""
    r = np.asarray(r)[:n_maps].astype(np.int64)
    if r.ndim == 1:
        r = r[:, None]
    u = (r + 0.5) / (N + 1)
    h = np.stack([(rbin[r] == b).sum(1) for b in range(NBIN)], 1)
    return np.concatenate([h, u.sum(1, keepdims=True), np.abs(u - 0.5).sum(1, keepdims=True),
                           ((r >= k50) & (r <= N - k50)).sum(1, keepdims=True),
                           ((r >= k90) & (r <= N - k90)).sum(1, keepdims=True), np.full((len(r), 1), r.shape[1])], 1)


def rank_metrics(tot):
    h, n = tot[:NBIN], tot[-1]
    f = h / n
    return np.array([np.sqrt(np.mean((f / p_bin - 1) ** 2)), tot[NBIN] / n - 0.5, 4 * tot[NBIN + 1] / n,
                     tot[NBIN + 2] / n - exact50, tot[NBIN + 3] / n - exact90])


def rand_rank(Ts, Tx, seed):
    """Randomised rank of truth among samples (handles ties for discrete statistics)."""
    Ts, Tx = np.asarray(Ts)[:n_maps], np.asarray(Tx)[:n_maps]
    lt = (Ts < Tx[:, None]).sum(1)
    eq = (Ts == Tx[:, None]).sum(1)
    return lt + np.random.default_rng(seed).integers(0, eq + 1)


# ---------------------------------------------------------------- map-level scalar statistics
PEAK_EDGES = np.round(np.arange(-0.1, 0.4 + 1e-9, 0.001), 4)
sig_ref = {}
for tag in ["2", "1"]:
    v = np.asarray(data[MOD[0]][1][f"mom{tag}_x"])[:n_maps, 0]
    sig_ref[tag] = float(np.sqrt(v.mean()))
RES["sigma_ref_smoothed_truth"] = sig_ref
NUS = [0, 1, 2, 3, 4]


def peak_counts_above(h, tag, nu):
    """h (..., nbins): counts of peaks with kappa_s > edge nearest nu*sigma_ref."""
    j = int(np.argmin(np.abs(PEAK_EDGES - nu * sig_ref[tag])))
    return np.asarray(h)[..., j + 1:].sum(-1), float(PEAK_EDGES[j])


def scalar_stats(arr):
    """dict name -> (samples (n, N), truth (n,))."""
    S = {}
    for b in range(nb):
        S[f"Pmask_l{ell[b]:.0f}"] = (arr["bp_mask_s"][:n_maps, :, b], arr["bp_mask_x"][:n_maps, b])
    for b in range(nb):
        S[f"Pfull_l{ell[b]:.0f}"] = (arr["bp_full_s"][:n_maps, :, b], arr["bp_full_x"][:n_maps, b])
    for tag in ["2", "1"]:
        hs, hx = np.asarray(arr[f"peaks{tag}_s"][:n_maps]), np.asarray(arr[f"peaks{tag}_x"][:n_maps])
        S[f"npeaks{tag}_all"] = (hs.sum(-1), hx.sum(-1))
        for nu in NUS:
            S[f"npeaks{tag}_nu>{nu}"] = (peak_counts_above(hs, tag, nu)[0], peak_counts_above(hx, tag, nu)[0])
    for nm, key in [("raw", ""), ("sm2", "2"), ("sm1", "1")]:
        for j, mo in enumerate(["var", "skew", "kurt"]):
            S[f"{mo}_{nm}"] = (arr[f"mom{key}_s"][:n_maps, :, j], arr[f"mom{key}_x"][:n_maps, j])
    S["mean_footprint"] = (arr["gm_s"][:n_maps], arr["gm_x"][:n_maps])
    return S


calib = {}
SC = {k: scalar_stats(a) for k, (_, a) in data.items()}
stat_names = list(SC[MOD[0]].keys())
for si, nm in enumerate(stat_names):
    kv = {k: boot(rank_agg(rand_rank(*SC[k][nm], seed=1000 + si)), rank_metrics) for k in MOD}
    calib[nm] = summ(kv)
TILE_STATS = [f"pm{s}" for s in [1, 2, 4, 8, 16, 32, 64]] + [f"pmax{s}" for s in [16, 32, 64]]
for nm in TILE_STATS:
    kv = {k: boot(rank_agg(a[f"{nm}_rank"]), rank_metrics) for k, (_, a) in data.items()}
    calib[nm] = summ(kv)
RES["rank_calibration"] = {"metrics": ["CE_rms_density_dev", "bias_meanPIT-0.5", "dispersion_4E|u-.5|",
                                       "cov50-exact", "cov90-exact"], "exact50": exact50, "exact90": exact90,
                           "CE_floor_single_stat": float(np.sqrt(np.mean((1 - p_bin) / (n_maps * p_bin)))), "stats": calib}
# chi2 uniformity p-values for map-level statistics (maps independent)
chi2p = {}
for si, nm in enumerate(stat_names):
    chi2p[nm] = {}
    for k in MOD:
        h = np.bincount(rbin[rand_rank(*SC[k][nm], seed=1000 + si)], minlength=NBIN)
        chi2p[nm][k] = float(sst.chisquare(h, p_bin * h.sum()).pvalue)
RES["rank_chi2_pvalue"] = chi2p

# ---------------------------------------------------------------- region-level coverage (A4)
LEV = [0.5, 0.8, 0.9, 0.95]
kq = [max(1, int(round((N + 1) * (1 - L) / 2))) for L in LEV]
exq = [(N + 1 - 2 * k) / (N + 1) for k in kq]
zc = [sst.norm.ppf((1 + L) / 2) for L in LEV]


def cov_agg(r, z):
    r, z = np.asarray(r)[:n_maps], np.asarray(z)[:n_maps]
    if r.ndim == 1:
        r, z = r[:, None], z[:, None]
    q = [((r >= k) & (r <= N - k)).sum(1) for k in kq]
    g = [(np.abs(z) <= c).sum(1) for c in zc]
    return np.stack(q + g + [(z.astype(float) ** 2).sum(1), np.full(len(r), r.shape[1])], 1)


def cov_metrics(t):
    n = t[-1]
    return np.array([t[i] / n - exq[i] for i in range(4)] + [t[4 + i] / n - LEV[i] for i in range(4)] + [t[8] / n / (1 + 1 / N)])


region = {}
for nm in [f"pm{s}" for s in [1, 2, 4, 8, 16, 32, 64]] + ["global"] + [f"pmax{s}" for s in [16, 32, 64]]:
    kv = {}
    for k, (_, a) in data.items():
        if nm == "global":
            gs, gx = np.asarray(a["gm_s"][:n_maps]), np.asarray(a["gm_x"][:n_maps])
            r = (gs < gx[:, None]).sum(1)
            z = (gx - gs.mean(1)) / gs.std(1, ddof=1)
        else:
            r, z = a[f"{nm}_rank"], a[f"{nm}_z"]
        kv[k] = boot(cov_agg(r, z), cov_metrics)
    region[nm] = summ(kv)
RES["region_coverage"] = {"metrics": [f"quantile_cov{L}-exact" for L in LEV] + [f"meanSD_cov{L}-nominal" for L in LEV] + ["mean_z2/(1+1/N)"],
                          "stats": region}

# ---------------------------------------------------------------- multivariate scores (A5)
VS_LAGS = ["(0,1)", "(1,0)", "(1,1)", "(1,-1)", "(0,2)", "(2,0)", "(0,4)", "(4,0)", "(0,8)", "(8,0)", "(0,16)", "(16,0)"]
scores = {}
for nm in ["es_full", "es_tile32", "es_down4"]:
    scores[nm] = summ({k: boot(np.asarray(a[nm])[:n_maps, None], lambda t: t[0] / n_maps) for k, (_, a) in data.items()})
vs_groups = {"vs_local": [0, 1, 2, 3], "vs_lag2": [4, 5], "vs_lag4": [6, 7], "vs_lag8": [8, 9], "vs_lag16": [10, 11]}
for nm, idx in vs_groups.items():
    scores[nm] = summ({k: boot(np.asarray(a["vs"])[:n_maps][:, idx].mean(1, keepdims=True), lambda t: t[0] / n_maps)
                       for k, (_, a) in data.items()})
RES["scores"] = scores

# ---------------------------------------------------------------- spectra (A1, B3)
spec = {}


def sp_ratio(num, den):
    return {k: boot(np.stack([np.asarray(num(a)), np.asarray(den(a))], 1), lambda t: t[0] / t[1]) for k, (_, a) in data.items()}


spec["mean_sample_over_truth_mask"] = {}
spec["postmean_over_truth_mask"] = {}
spec["spread_skill_err_over_spread"] = {}
spec["corr_mean_truth"] = {}
spec["mean_sample_over_truth_full"] = {}
spec["postmean_over_truth_full"] = {}
for b in range(nb):
    lab = f"l{ell[b]:.0f}"
    spec["mean_sample_over_truth_mask"][lab] = summ(sp_ratio(lambda a: a["bp_mask_s"][:n_maps, :, b].mean(1), lambda a: a["bp_mask_x"][:n_maps, b]))
    spec["postmean_over_truth_mask"][lab] = summ(sp_ratio(lambda a: a["bp_mask_m"][:n_maps, b], lambda a: a["bp_mask_x"][:n_maps, b]))
    spec["mean_sample_over_truth_full"][lab] = summ(sp_ratio(lambda a: a["bp_full_s"][:n_maps, :, b].mean(1), lambda a: a["bp_full_x"][:n_maps, b]))
    spec["postmean_over_truth_full"][lab] = summ(sp_ratio(lambda a: a["bp_full_m"][:n_maps, b], lambda a: a["bp_full_x"][:n_maps, b]))
    spec["spread_skill_err_over_spread"][lab] = summ(sp_ratio(lambda a: a["bp_res_x"][:n_maps, b] / (1 + 1 / N),
                                                              lambda a: a["bp_res_s"][:n_maps, b] / (1 - 1 / N)))
    spec["corr_mean_truth"][lab] = summ({k: boot(np.stack([a["bp_mask_cross_mx"][:n_maps, b], a["bp_mask_m"][:n_maps, b], a["bp_mask_x"][:n_maps, b]], 1),
                                                 lambda t: t[0] / np.sqrt(t[1] * t[2])) for k, (_, a) in data.items()})
# B3: per-sample spectra at high ell: distribution of P_i / P_truth over samples
per_samp = {}
HI = [b for b in range(nb) if k_eff[b] >= 32]
for k, (_, a) in data.items():
    rat = np.asarray(a["bp_mask_s"][:n_maps][:, :, HI]).sum(-1) / np.asarray(a["bp_mask_x"][:n_maps][:, HI]).sum(-1)[:, None]
    per_samp[k] = {"median": float(np.median(rat)), "p05": float(np.percentile(rat, 5)), "p95": float(np.percentile(rat, 95)),
                   "frac_below_0.8": float((rat < 0.8).mean()), "frac_below_0.5": float((rat < 0.5).mean()),
                   "within_map_cv": float(np.mean(rat.std(1) / rat.mean(1)))}
spec["per_sample_highell_ratio"] = {"bins_k>=32_ell>=": float(ell[HI[0]]), **per_samp}
RES["spectra"] = spec

# ---------------------------------------------------------------- spread-skill (B1)
ssk = {}
ssk["pixel_err2_over_var"] = summ({k: boot(np.stack([((np.asarray(a["pix_x"][:n_maps]) - a["pix_m"][:n_maps]) ** 2).sum(1) / (1 + 1 / N),
                                                     (np.asarray(a["pix_s"][:n_maps], dtype=float) ** 2).sum(1)], 1), lambda t: t[0] / t[1])
                                   for k, (_, a) in data.items()})
dec = {}
for k, (_, a) in data.items():
    m = np.asarray(a["pix_m"][:n_maps])
    edges = np.percentile(m[::7], np.linspace(0, 100, 11))
    di = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, 9)
    e2 = (np.asarray(a["pix_x"][:n_maps]) - m) ** 2 / (1 + 1 / N)
    v = np.asarray(a["pix_s"][:n_maps], dtype=float) ** 2
    agg = np.stack([np.stack([(e2 * (di == d)).sum(1), (v * (di == d)).sum(1)], 1) for d in range(10)], 1)  # (n, 10, 2)
    dec[k] = boot(agg, lambda t: t[:, 0] / t[:, 1])
ssk["err2_over_var_by_mhat_decile"] = summ(dec)
ssk["by_patch_size_meanz2"] = {nm: {k: {f: (v[8] if isinstance(v, list) else v) for f, v in region[nm][k].items()} for k in MOD} for nm in region}
# law of total variance across maps (per pixel, summed over footprint)


def ltv(k, a):
    x = np.asarray(a["pix_x"][:n_maps], dtype=float)
    m = np.asarray(a["pix_m"][:n_maps], dtype=float)
    s2 = np.asarray(a["pix_s"][:n_maps], dtype=float) ** 2
    out = []
    for W in [np.ones((1, n_maps))] + [C[i:i + 200] for i in range(0, args.B, 200)]:
        w = W / W.sum(1, keepdims=True)
        vx = w @ x ** 2 - (w @ x) ** 2
        vm = w @ m ** 2 - (w @ m) ** 2
        es2 = w @ s2
        out.append(np.stack([es2.sum(1) / vx.sum(1), vm.sum(1) / vx.sum(1), (es2 + vm).sum(1) / vx.sum(1)], 1))
    o = np.concatenate(out)
    return o[0], o[1:]


ssk["law_total_variance"] = {"metrics": ["E[s2]/Var(x)", "Var(m)/Var(x)", "(E[s2]+Var(m))/Var(x)"], **summ({k: ltv(k, a) for k, (_, a) in data.items()})}
RES["spread_skill"] = ssk

# ---------------------------------------------------------------- diversity (B2)
div = {}
off = ~np.eye(N, dtype=bool)
norm_c = np.sqrt(2 / (1 + 1 / N))
per_map_rho = {}
for k, (_, a) in data.items():
    D = np.asarray(a["dist"][:n_maps], dtype=float)
    dbar = D[:, off].mean(1)
    em = np.asarray(a["err_mean"][:n_maps])
    e = np.asarray(a["err_samp"][:n_maps], dtype=float)
    Dn = D + np.where(off, 0, np.inf)[None]
    nn_s = Dn.min(2).mean(1)  # mean over i of NN distance among the other N-1 samples
    # truth: NN among N-1 samples, leaving out sample i in turn
    srt = np.sort(e, 1)
    nn_x = np.where(e == srt[:, :1], srt[:, 1:2], srt[:, :1]).mean(1)
    close = (D[:, off] < 0.5 * dbar[:, None]).mean(1)
    rho = dbar / (norm_c * em)
    per_map_rho[k] = rho
    div[k] = {"pairdist_over_truthdist": boot(np.stack([dbar, norm_c * em], 1), lambda t: t[0] / t[1]),
              "nn_samples_over_nn_truth": boot(np.stack([nn_s, nn_x], 1), lambda t: t[0] / t[1]),
              "frac_pairs_closer_than_half_mean": boot(close[:, None], lambda t: t[0] / n_maps),
              "frac_maps_rho<0.5": boot((rho < 0.5)[:, None].astype(float), lambda t: t[0] / n_maps),
              "frac_maps_rho<0.1": boot((rho < 0.1)[:, None].astype(float), lambda t: t[0] / n_maps),
              "min_pair_over_mean_pair": boot((D[:, off].min(1) / dbar)[:, None], lambda t: t[0] / n_maps)}
RES["diversity"] = {m: summ({k: div[k][m] for k in MOD}) for m in div[MOD[0]]}
RES["diversity"]["median_rho"] = {k: float(np.median(per_map_rho[k])) for k in MOD}

# ---------------------------------------------------------------- epochs / BN (B4)
ep = {}
for f in sorted(glob.glob(os.path.join(OUT, "epochs", "*.npz"))):
    z = np.load(f)
    nm = os.path.basename(f)[:-4]
    n_e, Ne = len(z["v_sum"]), int(z["N"])
    ke = z["k_eff"]
    hi = ke >= 32
    rng_e = np.random.default_rng(0)
    Ce = np.stack([np.bincount(rng_e.integers(0, n_e, n_e), minlength=n_e) for _ in range(args.B)]).astype(float)

    def be(cols, g):
        a = np.stack(cols, 1)
        return g(a.sum(0)), np.array([g(d) for d in Ce @ a])

    cov = z["cov"]
    rows = {"err2_over_var": be([z["e_sum"] / (1 + 1 / Ne), z["v_sum"]], lambda t: t[0] / t[1]),
            "pairdist_over_truthdist": be([z["dbar"], np.sqrt(2 / (1 + 1 / Ne)) * z["err_mean"]], lambda t: t[0] / t[1]),
            "nn_ratio_over_meanpair": be([z["nn_ratio"]], lambda t: t[0] / n_e),
            "highell_sample_over_truth": be([z["bp_s"][:, hi].sum(1), z["bp_x"][:, hi].sum(1)], lambda t: t[0] / t[1]),
            "highell_postmean_over_truth": be([z["bp_m"][:, hi].sum(1), z["bp_x"][:, hi].sum(1)], lambda t: t[0] / t[1]),
            "highell_err_over_spread": be([z["res_x"][:, hi].sum(1) / (1 + 1 / Ne), z["res_s"][:, hi].sum(1) / (1 - 1 / Ne)], lambda t: t[0] / t[1]),
            "patch16_err2_over_var": be([z["e16"] / (1 + 1 / Ne), z["v16"]], lambda t: t[0] / t[1]),
            "pix_cov50-exact": be([cov[:, 0, 0] - cov[:, 0, 1]], lambda t: t[0] / n_e),
            "pix_cov90-exact": be([cov[:, 1, 0] - cov[:, 1, 1]], lambda t: t[0] / n_e)}
    ep[nm] = {r: {"val": float(v), "lo": float(ci(d)[0]), "hi": float(ci(d)[1])} for r, (v, d) in rows.items()}
RES["epochs"] = ep

# ---------------------------------------------------------------- latent sensitivity (B5)
lat = {}
for k in MOD:
    f = os.path.join(OUT, "latent", f"{k}.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    dG, dz, eps = z["dG"], z["dz"], z["eps"]
    nl = dG.shape[0]
    rng_l = np.random.default_rng(0)
    Cl = np.stack([np.bincount(rng_l.integers(0, nl, nl), minlength=nl) for _ in range(args.B)]).astype(float)
    g_map, z_map = dG.mean(1), dz.mean(1)  # (n, n_eps)
    rel = (g_map / g_map[:, -1:]) .mean(0)
    rel_d = np.array([((c @ g_map) / (c @ g_map[:, -1:]))  for c in Cl])  # ratio of means
    rel_v = g_map.sum(0) / g_map[:, -1].sum()
    sens = g_map.sum(0) / z_map.sum(0)
    sens_d = np.array([(c @ g_map) / (c @ z_map) for c in Cl])
    lat[k] = {"eps": eps.tolist(), "relative_change": rel_v.tolist(), "relative_change_lo": ci(rel_d)[0].tolist(),
              "relative_change_hi": ci(rel_d)[1].tolist(), "kappa_per_unit_z": sens.tolist(),
              "kappa_per_unit_z_lo": ci(sens_d)[0].tolist(), "kappa_per_unit_z_hi": ci(sens_d)[1].tolist(),
              "mean_of_per_map_relative": rel.tolist(), "_draws_rel": rel_d, "_draws_sens": sens_d}
if "R" in lat:
    for k in lat:
        if k != "R":
            for nm in ["rel", "sens"]:
                dd = lat[k][f"_draws_{nm}"] - lat["R"][f"_draws_{nm}"]
                lat[k][f"{nm}_minus_R_lo"], lat[k][f"{nm}_minus_R_hi"] = ci(dd)[0].tolist(), ci(dd)[1].tolist()
for k in lat:
    lat[k].pop("_draws_rel"), lat[k].pop("_draws_sens")
RES["latent"] = lat


# ---------------------------------------------------------------- write json
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


json.dump(clean(RES), open(os.path.join(AN, "joint.json"), "w"), indent=1)

# ---------------------------------------------------------------- text summary
L = []
L.append(f"n_maps={n_maps} N={N} B={args.B}; sigma_ref (smoothed truth rms) = {sig_ref}")
L.append("\n== Rank calibration: CE [95% CI] | CE(model)-CE(R) | dispersion | cov90-exact (pp)")
L.append(f"   CE floor for one map-level statistic (perfect calibration, n={n_maps}): {RES['rank_calibration']['CE_floor_single_stat']:.3f}")
for nm in stat_names + TILE_STATS:
    L.append(f"-- {nm}")
    for k in MOD:
        e = calib[nm][k]
        s = (f"   {k:5s} CE {e['val'][0]:.3f} [{e['lo'][0]:.3f},{e['hi'][0]:.3f}]  bias {e['val'][1]:+.3f}  disp {e['val'][2]:.3f} "
             f"[{e['lo'][2]:.3f},{e['hi'][2]:.3f}]  cov50 {100 * e['val'][3]:+.1f}  cov90 {100 * e['val'][4]:+.1f} [{100 * e['lo'][4]:+.1f},{100 * e['hi'][4]:+.1f}]")
        if k != "R":
            s += f"  dCE {e['minus_R'][0]:+.3f} [{e['minus_R_lo'][0]:+.3f},{e['minus_R_hi'][0]:+.3f}]"
        if nm in chi2p:
            s += f"  chi2p {chi2p[nm][k]:.2g}"
        L.append(s)
L.append("\n== Region coverage: quantile cov - exact (pp) at 50/80/90/95 | meanSD cov - nominal (pp) | mean z^2")
for nm in region:
    L.append(f"-- {nm}")
    for k in MOD:
        e = region[nm][k]
        v = e["val"]
        L.append(f"   {k:5s} q: " + " ".join(f"{100 * v[i]:+.1f}" for i in range(4)) + "  | g: " + " ".join(f"{100 * v[4 + i]:+.1f}" for i in range(4))
                 + f"  | z2 {v[8]:.3f} [{e['lo'][8]:.3f},{e['hi'][8]:.3f}]"
                 + (f"  q90-R {100 * e['minus_R'][2]:+.1f} [{100 * e['minus_R_lo'][2]:+.1f},{100 * e['minus_R_hi'][2]:+.1f}]" if k != "R" else ""))
L.append("\n== Scores (lower is better); model - R with 95% CI")
for nm in scores:
    for k in MOD:
        e = scores[nm][k]
        L.append(f"   {nm:10s} {k:5s} {e['val']:.6g} [{e['lo']:.6g},{e['hi']:.6g}]" + (f"  diff {e['minus_R']:+.4g} [{e['minus_R_lo']:+.4g},{e['minus_R_hi']:+.4g}]" if k != "R" else ""))
L.append("\n== Spectra by ell: <P_samp>/P_truth | P_mean/P_truth | err/spread | r(mean,truth)  (masked)")
for b in range(nb):
    lab = f"l{ell[b]:.0f}"
    s = f"   ell {ell[b]:7.0f}: "
    for k in MOD:
        s += (f"{k} {spec['mean_sample_over_truth_mask'][lab][k]['val']:.3f}/{spec['postmean_over_truth_mask'][lab][k]['val']:.3f}/"
              f"{spec['spread_skill_err_over_spread'][lab][k]['val']:.2f}/{spec['corr_mean_truth'][lab][k]['val']:.2f}  ")
    L.append(s)
L.append("   F100-R <P_samp>/P_truth by ell: " + "  ".join(
    f"{ell[b]:.0f}:{spec['mean_sample_over_truth_mask'][f'l{ell[b]:.0f}']['F100']['minus_R']:+.3f}[{spec['mean_sample_over_truth_mask'][f'l{ell[b]:.0f}']['F100']['minus_R_lo']:+.3f},{spec['mean_sample_over_truth_mask'][f'l{ell[b]:.0f}']['F100']['minus_R_hi']:+.3f}]"
    for b in range(nb)) if "F100" in MOD else "")
L.append("   per-sample high-ell ratio: " + json.dumps(per_samp))
L.append("\n== Spread-skill err2/var")
for k in MOD:
    e = ssk["pixel_err2_over_var"][k]
    L.append(f"   {k:5s} pixel {fmt(e)}" + (f" diff {fmt(e, diff=True)}" if k != "R" else "") + "  deciles " +
             " ".join(f"{v:.2f}" for v in ssk["err2_over_var_by_mhat_decile"][k]["val"]))
    e = ssk["law_total_variance"][k]
    L.append(f"         LTV E[s2]/Var(x) {e['val'][0]:.3f} Var(m)/Var(x) {e['val'][1]:.3f} sum {e['val'][2]:.3f} [{e['lo'][2]:.3f},{e['hi'][2]:.3f}]")
L.append("\n== Diversity")
for m_ in div[MOD[0]]:
    for k in MOD:
        e = RES["diversity"][m_][k]
        L.append(f"   {m_:34s} {k:5s} {fmt(e, 4)}" + (f" diff {fmt(e, 4, True)}" if k != "R" else ""))
L.append("   median rho: " + json.dumps(RES["diversity"]["median_rho"]))
L.append("\n== Epochs / BN")
for nm in ep:
    L.append(f"   {nm:22s} " + "  ".join(f"{r}={v['val']:.3f}[{v['lo']:.3f},{v['hi']:.3f}]" for r, v in ep[nm].items()))
L.append("\n== Latent sensitivity")
for k in lat:
    L.append(f"   {k:5s} eps {lat[k]['eps']}\n         rel {np.round(lat[k]['relative_change'], 3).tolist()}\n         kappa/unit z {np.round(lat[k]['kappa_per_unit_z'], 5).tolist()}")
open(os.path.join(AN, "summary.txt"), "w").write("\n".join(L))
print("\n".join(L[:3]))

# ---------------------------------------------------------------- figures
def errbar_line(ax, xv, kv, metric_idx=None, label=None):
    for k in MOD:
        v = np.array([kv[x][k]["val"] if metric_idx is None else kv[x][k]["val"][metric_idx] for x in kv])
        lo = np.array([kv[x][k]["lo"] if metric_idx is None else kv[x][k]["lo"][metric_idx] for x in kv])
        hi = np.array([kv[x][k]["hi"] if metric_idx is None else kv[x][k]["hi"][metric_idx] for x in kv])
        ax.plot(xv, v, color=COL[k], label=k, marker="o", ms=3)
        ax.fill_between(xv, lo, hi, color=COL[k], alpha=0.2, lw=0)


# J1 spectra
fig, axs = plt.subplots(2, 2, figsize=(10, 7))
for ax, (nm, tt) in zip(axs.flat, [("mean_sample_over_truth_mask", "mean per-sample spectrum / truth"),
                                    ("postmean_over_truth_mask", "posterior-mean spectrum / truth"),
                                    ("spread_skill_err_over_spread", "error / spread spectrum (1 = calibrated)"),
                                    ("corr_mean_truth", "r(posterior mean, truth)")]):
    errbar_line(ax, ell, spec[nm])
    ax.set_xscale("log")
    ax.set_title(tt)
    ax.set_xlabel("ℓ")
    if nm != "corr_mean_truth":
        ax.axhline(1, color="#52514e", lw=1)
    if nm == "spread_skill_err_over_spread":
        ax.set_yscale("log")
axs[0, 0].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "J1_spectra.png"), dpi=130)
plt.close(fig)

# J2 peak height histograms (2 arcmin)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
nu_edges = np.arange(-2, 7.01, 0.5)
cent = 0.5 * (nu_edges[1:] + nu_edges[:-1])
for k, (_, a) in data.items():
    hs = np.asarray(a["peaks2_s"][:n_maps], dtype=float).mean(1)  # (n, nbins)
    hx = np.asarray(a["peaks2_x"][:n_maps], dtype=float)
    up = np.concatenate([[-np.inf], PEAK_EDGES]) / sig_ref["2"]  # upper edge of bin j is PEAK_EDGES[j]
    up = up[1:]
    up = np.concatenate([up, [np.inf]])
    which = np.digitize(up, nu_edges) - 1
    reb = lambda h: np.stack([h[:, which == j].sum(1) for j in range(len(cent))], 1)
    Hs, Hx = reb(hs), reb(hx)
    axs[0].plot(cent, Hs.mean(0), color=COL[k], label=k)
    if k == MOD[0]:
        axs[0].plot(cent, Hx.mean(0), color="#0b0b0b", ls="--", label="truth")
    axs[1].plot(cent, Hs.sum(0) / np.maximum(Hx.sum(0), 1), color=COL[k], label=k)
axs[0].set_yscale("log")
axs[0].set_title("peaks per map (θ_G = 2′)")
axs[1].set_title("mean sample peak count / truth")
axs[1].axhline(1, color="#52514e", lw=1)
axs[1].set_ylim(0, 3)
for ax in axs:
    ax.set_xlabel("ν = κ_s / σ_ref")
axs[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "J2_peaks.png"), dpi=130)
plt.close(fig)

# J3 rank histograms for a selection of statistics
sel = [stat_names[2], stat_names[7], stat_names[12], "npeaks2_nu>3", "var_raw", "skew_sm2", "kurt_sm2", "mean_footprint", "pm8", "pm64", "pmax32"]
fig, axs = plt.subplots(len(sel), 1, figsize=(10, 1.4 * len(sel)), sharex=True)
for ax, nm in zip(axs, sel):
    w = 0.8 / len(MOD)
    for j, k in enumerate(MOD):
        e = calib[nm][k]
        # recompute histogram from json-free path
        if nm in SC[k]:
            r = rand_rank(*SC[k][nm], seed=1000 + stat_names.index(nm))
            h = np.bincount(rbin[r.ravel()], minlength=NBIN)
        else:
            r = np.asarray(data[k][1][f"{nm}_rank"][:n_maps]).ravel()
            h = np.bincount(rbin[r], minlength=NBIN)
        ax.bar(np.arange(NBIN) + (j - len(MOD) / 2 + 0.5) * w, h / h.sum() / p_bin, width=w, color=COL[k], label=k)
    ax.axhline(1, color="#52514e", lw=1)
    ax.set_ylabel(nm, rotation=0, ha="right", fontsize=8)
    ax.set_ylim(0, 3)
axs[0].legend(ncol=len(MOD), frameon=False, fontsize=8, loc="upper center")
axs[-1].set_xlabel("rank decile of truth among samples")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "J3_rank_hists.png"), dpi=130)
plt.close(fig)

# J4 calibration error per statistic
allst = stat_names + TILE_STATS
fig, ax = plt.subplots(figsize=(14, 4))
for j, k in enumerate(MOD):
    v = [calib[nm][k]["val"][0] for nm in allst]
    ax.plot(np.arange(len(allst)), v, color=COL[k], marker="o", ms=3, lw=1, label=k)
ax.axhline(RES["rank_calibration"]["CE_floor_single_stat"], color="#52514e", lw=1, ls="--", label="noise floor (map-level)")
ax.set_xticks(np.arange(len(allst)))
ax.set_xticklabels(allst, rotation=90, fontsize=6)
ax.set_yscale("log")
ax.set_ylabel("rank-histogram CE")
ax.legend(frameon=False, ncol=6)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "J4_calibration_error.png"), dpi=130)
plt.close(fig)

# J5 region coverage vs patch size
ps = [1, 2, 4, 8, 16, 32, 64]
xs = ps + [300]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (idx, tt) in zip(axs, [(2, "quantile interval, 90%: cov − exact (pp)"), (6, "mean ± 1.645 SD: cov − 90% (pp)"), (8, "mean z² (1 = calibrated)")]):
    for k in MOD:
        v = [region[nm][k]["val"][idx] for nm in [f"pm{s}" for s in ps] + ["global"]]
        lo = [region[nm][k]["lo"][idx] for nm in [f"pm{s}" for s in ps] + ["global"]]
        hi = [region[nm][k]["hi"][idx] for nm in [f"pm{s}" for s in ps] + ["global"]]
        sc = 1 if idx == 8 else 100
        ax.plot(xs, sc * np.array(v), color=COL[k], marker="o", ms=3, label=k)
        ax.fill_between(xs, sc * np.array(lo), sc * np.array(hi), color=COL[k], alpha=0.2, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in ps] + ["all"])
    ax.set_xlabel("patch size (px; 'all' = footprint mean)")
    ax.set_title(tt, fontsize=9)
    ax.axhline(1 if idx == 8 else 0, color="#52514e", lw=1)
axs[2].set_yscale("log")
axs[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "J5_region_coverage.png"), dpi=130)
plt.close(fig)

# J6 spread-skill by mhat decile, diversity histogram
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for k in MOD:
    e = ssk["err2_over_var_by_mhat_decile"][k]
    axs[0].plot(np.arange(1, 11), e["val"], color=COL[k], marker="o", ms=3, label=k)
    axs[0].fill_between(np.arange(1, 11), e["lo"], e["hi"], color=COL[k], alpha=0.2, lw=0)
    axs[1].hist(per_map_rho[k], bins=np.linspace(0, 1.6, 65), color=COL[k], alpha=0.6, label=k)
axs[0].set_yscale("log")
axs[0].axhline(1, color="#52514e", lw=1)
axs[0].set_xlabel("decile of posterior mean κ̂")
axs[0].set_title("pixel err² / var")
axs[1].axvline(1, color="#52514e", lw=1)
axs[1].set_xlabel("ρ = mean pairwise dist / (√(2/(1+1/N)) ‖x − m‖)")
axs[1].set_title("per-map sample diversity")
axs[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "J6_spread_skill_diversity.png"), dpi=130)
plt.close(fig)

# J7 epochs
if ep:
    mets = ["err2_over_var", "pairdist_over_truthdist", "highell_sample_over_truth", "highell_err_over_spread", "patch16_err2_over_var", "pix_cov90-exact"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for ax, mt in zip(axs.flat, mets):
        for k in ["R", "F100"]:
            rows = sorted([(int(n.split("_e")[1].split("_")[0]), v[mt]) for n, v in ep.items() if n.startswith(k + "_e") and n.endswith("_eval")])
            if not rows:
                continue
            xe = [r[0] for r in rows]
            ax.plot(xe, [r[1]["val"] for r in rows], color=COL[k], marker="o", ms=3, label=k)
            ax.fill_between(xe, [r[1]["lo"] for r in rows], [r[1]["hi"] for r in rows], color=COL[k], alpha=0.2, lw=0)
            for bn, mk in [("eval", "*"), ("train", "x")]:
                nmb = f"{k}_best_{bn}"
                if nmb in ep:
                    ax.plot([], [])
                    ax.scatter([100 if bn == "train" else 101], [ep[nmb][mt]["val"]], color=COL[k], marker=mk, s=40)
        ax.set_title(mt + "  (*: best eval, x: best train-BN)", fontsize=8)
        ax.set_xlabel("epoch")
    axs[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "J7_epochs.png"), dpi=130)
    plt.close(fig)

# J8 latent
if lat:
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for k in lat:
        e = np.array(lat[k]["eps"])
        axs[0].plot(e, lat[k]["relative_change"], color=COL[k], marker="o", ms=3, label=k)
        axs[0].fill_between(e, lat[k]["relative_change_lo"], lat[k]["relative_change_hi"], color=COL[k], alpha=0.2, lw=0)
        axs[1].plot(e, lat[k]["kappa_per_unit_z"], color=COL[k], marker="o", ms=3, label=k)
    axs[0].plot([0.01, 1], [0.01, 1], color="#52514e", lw=1, ls="--")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("ε  (z' = √(1−ε²) z + ε w)")
    axs[0].set_title("‖G(z')−G(z)‖ / ‖G(w)−G(z)‖")
    axs[1].set_title("‖G(z')−G(z)‖ / ‖z'−z‖  (κ per unit z)")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "J8_latent.png"), dpi=130)
    plt.close(fig)

# J9 example maps
fig, axs = plt.subplots(len(MOD), 6, figsize=(13, 2.2 * len(MOD)), squeeze=False)
for r_, k in enumerate(MOD):
    f = os.path.join(OUT, args.sub, k, "fig_samples.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    msk = z["mask"]
    ims = [z["x0"], z["m0"], z["s0"]] + list(z["samp0"][:3])
    tts = ["truth", "posterior mean", "posterior SD", "sample 1", "sample 2", "sample 3"]
    vlim = np.percentile(z["x0"][msk], [1, 99.5])
    for c_, (im, tt) in enumerate(zip(ims, tts)):
        ax = axs[r_, c_]
        imm = np.where(msk, im, np.nan)
        ax.imshow(imm, cmap="viridis" if c_ != 2 else "magma", vmin=None if c_ == 2 else vlim[0], vmax=None if c_ == 2 else vlim[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if r_ == 0:
            ax.set_title(tt)
    axs[r_, 0].set_ylabel(k)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "J9_example_maps.png"), dpi=110)
plt.close(fig)
print("figures written to", FIG)
