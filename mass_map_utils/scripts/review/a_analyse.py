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


def deciles(a, n=10):
    sub = np.asarray(a).ravel()[:: max(1, a.size // 2_000_000)]
    return np.quantile(sub, np.linspace(0, 1, n + 1)[1:-1])


def moments(z):
    z = z[np.isfinite(z)]
    zc = z - z.mean()
    v = (zc ** 2).mean()
    return float((zc ** 3).mean() / v ** 1.5), float((zc ** 4).mean() / v ** 2)


R = {}
for key in args.models:
    meta, pix, maps = load(key)
    N, n_maps = meta["n_samp"], meta["n_maps"]
    rank = np.asarray(pix["rank"]).astype(np.int32)
    m, s, x = (np.asarray(pix[k]).astype(np.float64) for k in ["m", "s", "x"])
    res = {"exp": meta["exp"], "ckpt": meta["ckpt"], "N": N, "n_maps": n_maps}

    # ---- A1: rank histogram and order-statistic coverage
    hist = np.bincount(rank.ravel(), minlength=N + 1).astype(float)
    res["rank_hist"] = hist.tolist()
    cov_rows = []
    for L, exact, cov_map in coverage_from_ranks(rank, N):
        cov_rows.append({"level": float(L), "exact": exact, "cov": float(cov_map.mean()),
                         "se": float(cov_map.std(ddof=1) / np.sqrt(n_maps))})
    res["A1_cov"] = cov_rows
    ecp32 = maps["ecp32"]
    res["A1_ecp32"] = [{"level": float(L), "ecp": float(ecp32[:, j].mean()), "ref_n32": float(L * 31 / 33)}
                       for j, L in enumerate(maps["ecp_levels"])]
    q = hist / hist.sum() * (N + 1)
    res["A1_rank_tails"] = {"lowest_1pct": float(q[: (N + 1) // 100].mean()),
                            "highest_1pct": float(q[-((N + 1) // 100):].mean()),
                            "central_50pct": float(q[(N + 1) // 4: 3 * (N + 1) // 4].mean())}

    # ---- A2: stratified coverage (by kappa-hat decile, SD decile, edge distance)
    strata = {}
    m_edges, s_edges = deciles(m), deciles(s)
    for name, lab in [("kappa_hat_decile", np.digitize(m, m_edges)), ("sd_decile", np.digitize(s, s_edges)),
                      ("edge_dist_bin", np.broadcast_to(np.digitize(dist_edge, EDGE_BINS) - 1, m.shape))]:
        rows = []
        for b in range(int(lab.max()) + 1):
            sel = lab == b
            rb = rank[sel]
            covs = {f"cov{int(L * 100)}": float(c.mean()) - e
                    for L, e, c in coverage_from_ranks(rb[None, :], N, [0.5, 0.9, 0.95])}
            h20 = np.bincount((rb * 20) // (N + 1), minlength=20) / rb.size * 20
            rows.append({"bin": b, "n": int(sel.sum()), **covs, "upper_tail_5pct": float(h20[-1]),
                         "lower_tail_5pct": float(h20[0]), "hist20": h20.tolist()})
        strata[name] = rows
    strata["kappa_hat_edges"] = m_edges.tolist()
    strata["sd_edges"] = s_edges.tolist()
    strata["edge_bins_px"] = [str(e) for e in EDGE_BINS]
    res["A2"] = strata

    # ---- A3: shape of the generated marginals; standardised truth
    skew = np.asarray(pix["skew"]).astype(np.float64)
    kurt = np.asarray(pix["kurt"]).astype(np.float64) + 3  # Pearson kurtosis (Gaussian = 3)
    z = (x - m) / s
    zs, zk = moments(z.ravel())
    lab = np.digitize(m, m_edges)
    a3 = {"skew_median": float(np.median(skew)), "skew_iqr": np.percentile(skew, [25, 75]).tolist(),
          "kurt_median": float(np.median(kurt)), "kurt_iqr": np.percentile(kurt, [25, 75]).tolist(),
          "samples_pooled_skew_mean": float(skew.mean()), "samples_pooled_kurt_mean": float(kurt.mean()),
          "truth_z_skew": zs, "truth_z_kurt": zk, "by_kappa_hat_decile": []}
    for b in range(10):
        sel = lab == b
        bs, bk = moments(z[sel])
        a3["by_kappa_hat_decile"].append({"bin": b, "skew_median": float(np.median(skew[sel])),
                                          "kurt_median": float(np.median(kurt[sel])),
                                          "truth_z_skew": bs, "truth_z_kurt": bk})
    res["A3"] = a3

    # ---- A4: spread-skill per kappa-hat decile
    mae, gmd = np.asarray(pix["mae"]).astype(np.float64), np.asarray(pix["gmd"]).astype(np.float64)
    err2 = (x - m) ** 2
    var_pred = s ** 2 * (1 + 1 / N)  # E[(x - mean_N)^2] under a perfect sampler
    a4 = {"overall": {"err2_over_var": float(err2.mean() / var_pred.mean()), "l1_ratio": float(mae.mean() / gmd.mean())},
          "by_kappa_hat_decile": []}
    for b in range(10):
        sel = lab == b
        a4["by_kappa_hat_decile"].append({"bin": b, "err2_over_var": float(err2[sel].mean() / var_pred[sel].mean()),
                                          "l1_ratio": float(mae[sel].mean() / gmd[sel].mean())})
    res["A4"] = a4

    # ---- A5: per-map proper scores
    res["A5_map_crps"] = (mae - gmd / 2).mean(1).tolist()
    res["A5_map_es"] = maps["es_fair"].tolist()

    # ---- A6: RCPS with per-map losses, Hoeffding UCB over maps
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_maps)
    cal, test = perm[: n_maps // 2], perm[n_maps // 2:]
    edges = maps["rho_edges"]
    lam_grid = edges[1:-1]  # loss(lam) = frac(rho > lam), evaluated at bin edges
    P = maps["rho_hist"].sum(-1)[:, 0]
    a6 = []
    delta = 0.05
    for li, L in enumerate(maps["rcps_levels"]):
        cum = np.cumsum(maps["rho_hist"][:, li, :], axis=1)[:, :-1]  # count(rho <= edge_{k+1})
        loss = 1 - cum / P[:, None]  # (n_maps, n_lam)
        n = len(cal)
        ucb = loss[cal].mean(0) + np.sqrt(np.log(1 / delta) / (2 * n))
        alpha = 1 - L
        ok = ucb <= alpha
        suffix_ok = np.flip(np.logical_and.accumulate(np.flip(ok)))
        if suffix_ok.any():
            j = int(np.argmax(suffix_ok))
            lam_hat = float(lam_grid[j])
            test_cov = float(1 - loss[test, j].mean())
            width = float(2 * lam_hat * maps["half_width"][test, li].mean())
        else:
            lam_hat, test_cov, width = None, None, None
        j1 = int(np.searchsorted(lam_grid, 1.0))
        a6.append({"level": float(L), "lam_hat": lam_hat, "test_cov": test_cov, "width": width,
                   "raw_cal_cov": float(1 - loss[cal, j1].mean()), "raw_width": float(2 * maps["half_width"][test, li].mean()),
                   "hoeffding_penalty": float(np.sqrt(np.log(1 / delta) / (2 * n)))})
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

json.dump(R, open(OUT_ROOT + "analysis/stageA.json", "w"))

# ---- figures
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for key, res in R.items():
    h = np.array(res["rank_hist"])
    N = res["N"]
    h40 = np.add.reduceat(h, np.linspace(0, N + 1, 41).astype(int)[:-1])
    ax[0].plot(np.linspace(0, 1, 40), h40 / h40.sum() * 40, label=key, color=COLORS.get(key))
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
