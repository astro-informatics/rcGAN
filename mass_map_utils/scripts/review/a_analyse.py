"""Stage A analysis (CPU) of the per-pixel statistics written by a_sample_stats.py.

Writes OUT_ROOT/analysis/stageA.json and figures under OUT_ROOT/figures/.
Usage: python a_analyse.py [--models R F3 F100 FB S NL] [--ref F3]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from common import OUT_ROOT  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["R", "F3", "F100", "FB", "S", "NL"])
p.add_argument("--ref", default="F3")
p.add_argument("--stats-dir", default="stats")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

FIG = OUT_ROOT + "figures/"
os.makedirs(FIG, exist_ok=True)
os.makedirs(OUT_ROOT + "analysis", exist_ok=True)
TAG = "" if args.stats_dir == "stats" else "_" + args.stats_dir
FIG = FIG if not TAG else OUT_ROOT + f"figures/{args.stats_dir}/"
os.makedirs(FIG, exist_ok=True)
LEVELS = np.round(np.arange(0.05, 1.0, 0.05), 2)
COLORS = {"R": "#d62728", "F3": "#1f77b4", "F100": "#17becf", "FB": "#9467bd", "S": "#ff7f0e", "NL": "#7f7f7f"}
mask = np.load("/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_mask.npy", allow_pickle=True).astype(bool)
dist_edge = ndimage.distance_transform_edt(mask)[mask]  # px to nearest non-footprint pixel
EDGE_BINS = [0, 3, 6, 12, 24, 48, np.inf]


def load(key):
    d = f"{OUT_ROOT}{args.stats_dir}/{key}/"
    meta = json.load(open(d + "meta.json"))
    pix = {k: np.load(d + f"pix_{k}.npy", mmap_mode="r") for k in ["m", "s", "skew", "kurt", "x", "rank", "mae", "gmd"]}
    return meta, pix, dict(np.load(d + "maps.npz"))


def coverage_from_ranks(rank, N, levels=LEVELS):
    """Order-statistic central intervals [X_(k), X_(N+1-k)]: covered iff k <= rank <= N-k."""
    out = []
    for L in levels:
        k = max(1, int(round((N + 1) * (1 - L) / 2)))
        exact = (N + 1 - 2 * k) / (N + 1)
        cov = ((rank >= k) & (rank <= N - k)).mean(axis=-1)
        out.append((L, exact, cov))
    return out


def hb_pvalue(r_hat, alpha, n):
    """Hoeffding-Bentkus p-value for H0: R > alpha given empirical risk r_hat (Bates et al. 2021, eq. 13)."""
    from scipy.stats import binom

    r = np.minimum(r_hat, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = np.where(r > 0, r * np.log(r / alpha), 0.0) + (1 - r) * np.log((1 - r) / (1 - alpha))
    p_h = np.exp(-n * h1)
    p_b = np.e * binom.cdf(np.ceil(n * np.asarray(r_hat)), n, alpha)
    return np.minimum(p_h, p_b)


def deciles(a, n=10):
    sub = np.asarray(a).ravel()[:: max(1, a.size // 2_000_000)]
    return np.quantile(sub, np.linspace(0, 1, n + 1)[1:-1])


def moments(z):
    z = z[np.isfinite(z)]
    zc = z - z.mean()
    v = (zc ** 2).mean()
    return float((zc ** 3).mean() / v ** 1.5), float((zc ** 4).mean() / v ** 2)


CHUNK = 50  # maps per chunk; keeps memory to a few hundred MB
SUB = 4  # every SUB-th map is kept for quantiles (decile edges, skew/kurt medians/IQR)


class Moments:
    """Running raw power sums -> pooled skewness / Pearson kurtosis."""

    def __init__(self):
        self.p = np.zeros(5)

    def add(self, z):
        z = z[np.isfinite(z)]
        self.p += [z.size, z.sum(), (z ** 2).sum(), (z ** 3).sum(), (z ** 4).sum()]

    def get(self):
        n, s1, s2, s3, s4 = self.p
        mu = s1 / n
        m2 = s2 / n - mu ** 2
        m3 = s3 / n - 3 * mu * s2 / n + 2 * mu ** 3
        m4 = s4 / n - 4 * mu * s3 / n + 6 * mu ** 2 * s2 / n - 3 * mu ** 4
        return float(m3 / m2 ** 1.5), float(m4 / m2 ** 2)


R = {}
for key in args.models:
    meta, pix, maps = load(key)
    N, n_maps = meta["n_samp"], meta["n_maps"]
    res = {"exp": meta["exp"], "ckpt": meta["ckpt"], "N": N, "n_maps": n_maps}
    chunks = [slice(a, min(a + CHUNK, n_maps)) for a in range(0, n_maps, CHUNK)]
    f64 = lambda k, c: np.asarray(pix[k][c]).astype(np.float64)  # noqa: E731

    # pass 1: decile edges from a subsample of maps
    sub = slice(0, n_maps, SUB)
    m_edges, s_edges = deciles(np.asarray(pix["m"][sub])), deciles(np.asarray(pix["s"][sub]))
    skew_sub = np.asarray(pix["skew"][sub]).astype(np.float64)
    kurt_sub = np.asarray(pix["kurt"][sub]).astype(np.float64) + 3  # Pearson kurtosis (Gaussian = 3)
    lab_sub = np.digitize(np.asarray(pix["m"][sub]), m_edges)

    # pass 2: streamed accumulators
    hist = np.zeros(N + 1)
    cov_map = np.zeros((n_maps, len(LEVELS)))
    edge_lab = np.digitize(dist_edge, EDGE_BINS) - 1
    strata_hist = {"kappa_hat_decile": np.zeros((10, N + 1)), "sd_decile": np.zeros((10, N + 1)),
                   "edge_dist_bin": np.zeros((len(EDGE_BINS) - 1, N + 1))}
    z_all, z_dec = Moments(), [Moments() for _ in range(10)]
    a4_sums = np.zeros((10, 4))  # err2, var_pred, mae, gmd sums per kappa-hat decile
    a4_tot = np.zeros(4)
    map_crps = np.zeros(n_maps)
    for c in chunks:
        rank = np.asarray(pix["rank"][c]).astype(np.int64)
        m, s, x, mae, gmd = (f64(k, c) for k in ["m", "s", "x", "mae", "gmd"])
        hist += np.bincount(rank.ravel(), minlength=N + 1)
        for j, (L, exact, cm) in enumerate(coverage_from_ranks(rank, N)):
            cov_map[c, j] = cm
        labs = {"kappa_hat_decile": np.digitize(m, m_edges), "sd_decile": np.digitize(s, s_edges),
                "edge_dist_bin": np.broadcast_to(edge_lab, m.shape)}
        for name, lab in labs.items():
            nb = strata_hist[name].shape[0]
            strata_hist[name] += np.bincount((lab * (N + 1) + rank).ravel(), minlength=nb * (N + 1)).reshape(nb, N + 1)
        z = (x - m) / s
        z_all.add(z.ravel())
        err2, var_pred = (x - m) ** 2, s ** 2 * (1 + 1 / N)  # E[(x - mean_N)^2] = (1+1/N) var under a perfect sampler
        lab = labs["kappa_hat_decile"]
        for b in range(10):
            sel = lab == b
            z_dec[b].add(z[sel])
            a4_sums[b] += [err2[sel].sum(), var_pred[sel].sum(), mae[sel].sum(), gmd[sel].sum()]
        a4_tot += [err2.sum(), var_pred.sum(), mae.sum(), gmd.sum()]
        map_crps[c] = (mae - gmd / 2).mean(1)
    exact_levels = [e for _, e, _ in coverage_from_ranks(np.zeros((1, 1), int), N)]

    # ---- A1: rank histogram and order-statistic coverage
    res["rank_hist"] = hist.tolist()
    res["A1_cov"] = [{"level": float(L), "exact": exact_levels[j], "cov": float(cov_map[:, j].mean()),
                      "se": float(cov_map[:, j].std(ddof=1) / np.sqrt(n_maps))} for j, L in enumerate(LEVELS)]
    ecp32 = maps["ecp32"]
    res["A1_ecp32"] = [{"level": float(L), "ecp": float(ecp32[:, j].mean()), "ref_n32": float(L * 31 / 33),
                        "se": float(ecp32[:, j].std(ddof=1) / np.sqrt(n_maps))} for j, L in enumerate(maps["ecp_levels"])]
    q = hist / hist.sum() * (N + 1)
    res["A1_rank_tails"] = {"lowest_1pct": float(q[: max(1, (N + 1) // 100)].mean()),
                            "highest_1pct": float(q[-max(1, (N + 1) // 100):].mean()),
                            "central_50pct": float(q[(N + 1) // 4: 3 * (N + 1) // 4].mean())}

    # ---- A2: stratified rank histograms (never on the true kappa)
    strata = {}
    r_idx = np.arange(N + 1)
    for name, H in strata_hist.items():
        rows = []
        for b, h in enumerate(H):
            if h.sum() == 0:
                continue
            covs = {}
            for L in [0.5, 0.9, 0.95]:
                k = max(1, int(round((N + 1) * (1 - L) / 2)))
                covs[f"cov{int(L * 100)}"] = float(h[k:N - k + 1].sum() / h.sum()) - (N + 1 - 2 * k) / (N + 1)
            h20 = np.bincount((r_idx * 20) // (N + 1), weights=h, minlength=20) / h.sum() * 20
            rows.append({"bin": b, "n": int(h.sum()), **covs, "upper_tail_5pct": float(h20[-1]),
                         "lower_tail_5pct": float(h20[0]), "hist20": h20.tolist()})
        strata[name] = rows
    strata["kappa_hat_edges"] = m_edges.tolist()
    strata["sd_edges"] = s_edges.tolist()
    strata["edge_bins_px"] = [str(e) for e in EDGE_BINS]
    res["A2"] = strata

    # ---- A3: shape of the generated marginals (subsample of maps); pooled standardised truth (all maps)
    zs, zk = z_all.get()
    a3 = {"skew_median": float(np.median(skew_sub)), "skew_iqr": np.percentile(skew_sub, [25, 75]).tolist(),
          "kurt_median": float(np.median(kurt_sub)), "kurt_iqr": np.percentile(kurt_sub, [25, 75]).tolist(),
          "samples_pooled_skew_mean": float(skew_sub.mean()), "samples_pooled_kurt_mean": float(kurt_sub.mean()),
          "truth_z_skew": zs, "truth_z_kurt": zk, "by_kappa_hat_decile": []}
    for b in range(10):
        sel = lab_sub == b
        bs, bk = z_dec[b].get()
        a3["by_kappa_hat_decile"].append({"bin": b, "skew_median": float(np.median(skew_sub[sel])),
                                          "kurt_median": float(np.median(kurt_sub[sel])),
                                          "truth_z_skew": bs, "truth_z_kurt": bk})
    res["A3"] = a3

    # ---- A4: spread-skill per kappa-hat decile
    res["A4"] = {"overall": {"err2_over_var": float(a4_tot[0] / a4_tot[1]), "l1_ratio": float(a4_tot[2] / a4_tot[3])},
                 "by_kappa_hat_decile": [{"bin": b, "err2_over_var": float(r[0] / r[1]), "l1_ratio": float(r[2] / r[3])}
                                         for b, r in enumerate(a4_sums)]}

    # ---- A5: per-map proper scores
    res["A5_map_crps"] = map_crps.tolist()
    res["A5_map_es"] = maps["es_fair"].tolist()

    # ---- A6: RCPS with per-map losses (n = #calibration maps). Two bounds:
    #   naive Hoeffding UCB  R_hat + sqrt(log(1/delta)/(2n))   (as calibration.py, for continuity)
    #   Hoeffding-Bentkus (Bates et al. 2021): certify R(lam) <= alpha iff p_HB(R_hat, alpha) < delta,
    #   p_HB = min(exp(-n h1(min(R_hat, alpha), alpha)), e * P(Bin(n, alpha) <= ceil(n R_hat)))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_maps)
    cal, test = perm[: n_maps // 2], perm[n_maps // 2:]
    edges = maps["rho_edges"]
    lam_grid = edges[1:-1]  # loss(lam) = frac(rho > lam), evaluated at bin edges
    P = maps["rho_hist"].sum(-1)[:, 0]
    a6 = []
    delta = 0.05
    n = len(cal)
    for li, L in enumerate(maps["rcps_levels"]):
        cum = np.cumsum(maps["rho_hist"][:, li, :], axis=1)[:, :-1]  # count(rho <= edge_{k+1})
        loss = 1 - cum / P[:, None]  # (n_maps, n_lam), non-increasing in lam
        alpha = 1 - L
        r_hat = loss[cal].mean(0)
        row = {"level": float(L), "raw_cal_cov": float(1 - loss[cal, int(np.searchsorted(lam_grid, 1.0))].mean()),
               "raw_width": float(2 * maps["half_width"][test, li].mean()),
               "hoeffding_penalty": float(np.sqrt(np.log(1 / delta) / (2 * n)))}
        certs = {"hoeffding": r_hat + np.sqrt(np.log(1 / delta) / (2 * n)) <= alpha,
                 "HB": hb_pvalue(r_hat, alpha, n) < delta}
        for name, ok in certs.items():
            suffix_ok = np.flip(np.logical_and.accumulate(np.flip(ok)))
            if suffix_ok.any():
                j = int(np.argmax(suffix_ok))
                lam_hat = float(lam_grid[j])
                row[name] = {"lam_hat": lam_hat, "test_cov": float(1 - loss[test, j].mean()),
                             "width": float(2 * lam_hat * maps["half_width"][test, li].mean())}
            else:
                row[name] = {"lam_hat": None, "test_cov": None, "width": None, "note": "cannot certify on grid lam<=5"}
        a6.append(row)
    res["A6"] = a6
    R[key] = res
    print(key, "done", flush=True)

# ---- A5 paired comparison vs reference
ref = args.ref
if ref in R:
    for key in R:
        for sc in ["A5_map_crps", "A5_map_es"]:
            d = np.array(R[key][sc]) - np.array(R[ref][sc])
            rng = np.random.default_rng(1)
            boots = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(2000)]
            R[key][sc + "_vs_ref"] = {"ref": ref, "mean": float(np.mean(R[key][sc])), "diff": float(d.mean()),
                                      "se": float(d.std(ddof=1) / np.sqrt(d.size)),
                                      "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size))) if d.std() > 0 else 0.0,
                                      "boot95": np.percentile(boots, [2.5, 97.5]).tolist()}

json.dump(R, open(OUT_ROOT + f"analysis/stageA{TAG}.json", "w"))

# ---- figures
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for key, res in R.items():
    h = np.array(res["rank_hist"])
    N = res["N"]
    starts = np.linspace(0, N + 1, 41).astype(int)
    h40 = np.add.reduceat(h, starts[:-1]) / np.diff(starts)  # per-rank density (bins hold 6 or 7 ranks)
    ax[0].plot((starts[:-1] + starts[1:]) / 2 / (N + 1), h40 / h.mean(), label=key, color=COLORS.get(key))
    c = res["A1_cov"]
    ax[1].plot([r["exact"] for r in c], [r["cov"] - r["exact"] for r in c], "o-", label=key, color=COLORS.get(key))
ax[0].axhline(1, color="k", lw=0.5)
ax[0].set(xlabel="normalised rank of truth among N samples", ylabel="density", yscale="log", title="A1 rank histogram (pooled)")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set(xlabel="exact nominal level (N+1-2k)/(N+1)", ylabel="coverage - nominal", title="A1 coverage error")
ax[0].legend()
fig.tight_layout()
fig.savefig(FIG + "A1_rank_coverage.png", dpi=130)

for strat in ["kappa_hat_decile", "sd_decile", "edge_dist_bin"]:
    fig, axs = plt.subplots(1, len(R), figsize=(3.2 * len(R), 3), sharey=True)
    for a, (key, res) in zip(np.atleast_1d(axs), R.items()):
        rows = res["A2"][strat]
        img = np.array([r["hist20"] for r in rows])
        im = a.imshow(np.log2(img), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2, origin="lower")
        a.set(title=key, xlabel="rank bin (20)", ylabel=strat if a is np.atleast_1d(axs)[0] else None)
    fig.colorbar(im, ax=axs, label="log2 density")
    fig.savefig(FIG + f"A2_{strat}.png", dpi=130)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for key, res in R.items():
    rows = res["A3"]["by_kappa_hat_decile"]
    ax[0].plot([r["skew_median"] for r in rows], "o-", color=COLORS.get(key), label=f"{key} samples")
    ax[0].plot([r["truth_z_skew"] for r in rows], "x--", color=COLORS.get(key), label=f"{key} truth z")
    ax[1].plot([r["kurt_median"] for r in rows], "o-", color=COLORS.get(key))
    ax[1].plot([r["truth_z_kurt"] for r in rows], "x--", color=COLORS.get(key))
ax[0].set(xlabel="kappa-hat decile", ylabel="skewness", title="A3 skewness (o samples median, x pooled truth z)")
ax[1].set(xlabel="kappa-hat decile", ylabel="Pearson kurtosis", title="A3 kurtosis", yscale="log")
ax[0].legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(FIG + "A3_shape.png", dpi=130)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for key, res in R.items():
    rows = res["A4"]["by_kappa_hat_decile"]
    ax[0].plot([r["err2_over_var"] for r in rows], "o-", color=COLORS.get(key), label=key)
    ax[1].plot([r["l1_ratio"] for r in rows], "o-", color=COLORS.get(key), label=key)
for a, t in zip(ax, ["E(x-m)^2 / (1+1/N) s^2", "E|X-x| / E|X-X'|"]):
    a.axhline(1, color="k", lw=0.5)
    a.set(xlabel="kappa-hat decile", title="A4 " + t)
ax[0].legend()
fig.tight_layout()
fig.savefig(FIG + "A4_spread_skill.png", dpi=130)
print("figures written to", FIG)
