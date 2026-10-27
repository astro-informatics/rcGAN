"""CPU analysis of eval_sample.py outputs: bootstrap 95% CIs over test images (2000 resamples, seed 0,
same resampled indices for every model, so paired differences vs R get CIs), results.json and figures.

Usage: python eval_analyse.py R F MS [--ref R]
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "/share/gpu0/jjwhit/crps_review/paper/radio/eval/"
LEVELS = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
PATCH = [1, 2, 4, 8, 16, 32, 64, 128]
NB = 2000
COLORS = {"R": "#1f77b4", "F": "#d62728", "MS": "#2ca02c"}


def interval(N, a):
    """Order-statistic central interval [s_(l), s_(u)] and its exact coverage (u - l) / (N + 1)."""
    l = int(np.floor((1 - a) / 2 * (N + 1)))
    l = max(l, 1)
    u = N + 1 - l
    return l, u, (u - l) / (N + 1)


def coverage_from_hist(h, N):
    """h: (..., N+1) rank counts -> coverage at each level minus exact level, (..., len(LEVELS))."""
    tot = h.sum(-1)
    out = []
    for a in LEVELS:
        l, u, ex = interval(N, a)
        out.append(h[..., l:u].sum(-1) / np.maximum(tot, 1) - ex)
    return np.stack(out, -1)


def calib_err(h, nbins=10):
    """RMS deviation of a 10-bin rank-histogram density from 1."""
    N1 = h.shape[-1]
    edges = np.linspace(0, N1, nbins + 1).round().astype(int)
    b = np.stack([h[..., edges[j]:edges[j + 1]].sum(-1) for j in range(nbins)], -1).astype(float)
    w = np.diff(edges) / N1
    dens = b / np.maximum(b.sum(-1, keepdims=True), 1) / w
    return np.sqrt(((dens - 1) ** 2).mean(-1))


def pit_stats(h):
    """bias (mean PIT - 1/2) and dispersion 4 E|PIT - 1/2| (1 uniform, > 1 under-dispersed)."""
    N1 = h.shape[-1]
    u = (np.arange(N1) + 0.5) / N1
    p = h / np.maximum(h.sum(-1, keepdims=True), 1)
    return (p * u).sum(-1) - 0.5, 4 * (p * np.abs(u - 0.5)).sum(-1)


def boot(stat, n, idx):
    """stat: function of an index array -> np array. Returns point, lo, hi."""
    pt = np.asarray(stat(np.arange(n)))
    bs = np.stack([np.asarray(stat(ix)) for ix in idx])
    return pt, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), bs


def fmt(pt, lo, hi, d=3):
    return [round(float(v), d) for v in (pt, lo, hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--ref", default="R")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    S = {t: dict(np.load(f"{OUT}/stats_{t}.npz")) for t in a.tags}
    n = min(len(s["psnr"]) for s in S.values())
    N = int(S[a.tags[0]]["n_samp"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(NB, n))
    k_eff = S[a.tags[0]]["k_eff"]
    res = {"n_test": n, "n_samp": N, "levels": LEVELS, "patch": PATCH, "k_eff": k_eff.tolist(), "models": {}}

    # metric definitions: name -> function(stats dict) -> function(ix) -> value(s)
    def mean_of(key, col=None):
        return lambda s: (lambda ix: s[key][:n][ix].mean(0) if col is None else s[key][:n][ix, col].mean(0))

    def ratio(k1, k2, mult=1.0):
        return lambda s: (lambda ix: mult * np.nansum(s[k1][:n][ix]) / np.nansum(s[k2][:n][ix]))

    def cov(key, j=None):
        return lambda s: (lambda ix: coverage_from_hist((s[key][:n][ix] if j is None else s[key][:n][ix, j]).sum(0), N))

    def ce(key, j=None):
        return lambda s: (lambda ix: calib_err((s[key][:n][ix] if j is None else s[key][:n][ix, j]).sum(0)))

    def disp(key, j=None):
        return lambda s: (lambda ix: np.array(pit_stats((s[key][:n][ix] if j is None else s[key][:n][ix, j]).sum(0))))

    def pow_ratio(k1, k2):
        return lambda s: (lambda ix: s[k1][:n][ix].sum(0) / s[k2][:n][ix].sum(0))

    def bandrank_ce():
        def f(s):
            r = s["pow_rank"][:n]  # (n, nb)
            def g(ix):
                h = np.stack([np.bincount(r[ix, b], minlength=N + 1) for b in range(r.shape[1])])
                return calib_err(h)
            return g
        return f

    def bandrank_cov90():
        def f(s):
            r = s["pow_rank"][:n]
            l, u, ex = interval(N, 0.9)
            return lambda ix: ((r[ix] >= l) & (r[ix] <= u - 1)).mean(0) - ex
        return f

    metrics = {
        "psnr_P1": mean_of("psnr", 0), "psnr_P8": mean_of("psnr", 1), "psnr_P32": mean_of("psnr", 2),
        "ssim_P1": mean_of("ssim", 0), "ssim_P8": mean_of("ssim", 1), "ssim_P32": mean_of("ssim", 2),
        "crps_fair": mean_of("crps"), "ms_crps_fair": mean_of("mscrps"), "energy_score": mean_of("es"),
        "variogram_local": lambda s: (lambda ix: s["vs"][:n][ix].mean()),
        "spread_skill_v_over_e": ratio("v", "e", 1.0 + 1.0 / N),
        "spread_skill_src_v_over_e": ratio("v_pix_src", "e_pix_src", 1.0 + 1.0 / N),
        "pairdist_ratio": ratio("pair_d", "truth_d"),
        "near_identical_frac_1pct": mean_of("near_ident", 0), "near_identical_frac_5pct": mean_of("near_ident", 1),
        "near_identical_frac_10pct": mean_of("near_ident", 2),
        "cov_pix_all": cov("rank_pix_all"), "cov_pix_src": cov("rank_pix_src"), "cov_pix_bg": cov("rank_pix_bg"),
        "cov_pix_truthmask": cov("rank_pix_tru"),
        "calerr_pix_all": ce("rank_pix_all"), "calerr_pix_src": ce("rank_pix_src"), "calerr_pix_bg": ce("rank_pix_bg"),
        "pit_bias_disp_pix_all": disp("rank_pix_all"), "pit_bias_disp_pix_src": disp("rank_pix_src"),
        "pow_ratio_sample_over_truth": pow_ratio("pow_samp", "pow_true"),
        "pow_ratio_mean_over_truth": pow_ratio("pow_mean", "pow_true"),
        "pow_ratio_cross_over_truth": pow_ratio("pow_cross", "pow_true"),
        "bandpower_rank_calerr": bandrank_ce(), "bandpower_cov90_minus_exact": bandrank_cov90(),
    }
    for j, s_ in enumerate(PATCH):
        metrics[f"cov_patch{s_}"] = cov("patch_rank", j)
        metrics[f"cov_patch{s_}_src"] = cov("patch_src_rank", j)
        metrics[f"calerr_patch{s_}"] = ce("patch_rank", j)
        metrics[f"pit_bias_disp_patch{s_}"] = disp("patch_rank", j)

    boots = {}
    for t, s in S.items():
        res["models"][t] = {}
        boots[t] = {}
        for name, mk in metrics.items():
            pt, lo, hi, bs = boot(mk(s), n, idx)
            boots[t][name] = (pt, bs)
            res["models"][t][name] = {"est": np.round(pt, 5).tolist(), "lo": np.round(lo, 5).tolist(),
                                      "hi": np.round(hi, 5).tolist()}
        res["models"][t]["sec_per_image"] = float(np.mean(s["time"]))
        res["models"][t]["mask_frac_src_truth_recall"] = np.nanmean(s["mask_frac"][:n], 0).round(4).tolist()
    if a.ref in S:
        res["paired_diff_vs_" + a.ref] = {}
        for t in S:
            if t == a.ref:
                continue
            dd = {}
            for name in metrics:
                p1, b1 = boots[t][name]
                p0, b0 = boots[a.ref][name]
                db = b1 - b0
                dd[name] = {"est": np.round(p1 - p0, 5).tolist(), "lo": np.round(np.percentile(db, 2.5, 0), 5).tolist(),
                            "hi": np.round(np.percentile(db, 97.5, 0), 5).tolist()}
            res["paired_diff_vs_" + a.ref][t] = dd
    with open(f"{OUT}/results{a.suffix}.json", "w") as f:
        json.dump(res, f, indent=1)
    print_tables(res, a.tags)
    figures(res, S, a.tags, n, N, a.suffix)


def print_tables(res, tags):
    def row(name, j=None, d=3):
        out = []
        for t in tags:
            m = res["models"][t][name]
            e, l, h = (m["est"], m["lo"], m["hi"]) if j is None else (m["est"][j], m["lo"][j], m["hi"][j])
            out.append(f"{e:.{d}f} [{l:.{d}f}, {h:.{d}f}]")
        return " | ".join(out)
    print("metric | " + " | ".join(tags))
    for nm in ["psnr_P1", "psnr_P8", "psnr_P32", "ssim_P1", "ssim_P8", "ssim_P32", "crps_fair", "ms_crps_fair",
               "energy_score", "variogram_local", "spread_skill_v_over_e", "spread_skill_src_v_over_e", "pairdist_ratio",
               "near_identical_frac_5pct", "calerr_pix_all", "calerr_pix_src", "bandpower_rank_calerr"]:
        print(nm, "|", row(nm, d=4))
    for nm in ["cov_pix_all", "cov_pix_src", "cov_pix_bg"] + [f"cov_patch{s}" for s in PATCH]:
        for j, lv in enumerate(LEVELS):
            if lv in (0.5, 0.9):
                print(nm, lv, "|", row(nm, j))


def figures(res, S, tags, n, N, suffix):
    K = np.array(res["k_eff"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for t in tags:
        m = res["models"][t]
        for a_, key, ttl in [(ax[0], "cov_pix_all", "per-pixel, all"), (ax[1], "cov_pix_src", "per-pixel, source mask")]:
            e, lo, hi = (np.array(m[key][q]) for q in ("est", "lo", "hi"))
            a_.errorbar(LEVELS, e, yerr=[e - lo, hi - e], marker="o", capsize=3, label=t, color=COLORS.get(t))
            a_.set_title(ttl)
        j90 = LEVELS.index(0.9)
        e = np.array([m[f"cov_patch{s}"]["est"][j90] for s in PATCH])
        lo = np.array([m[f"cov_patch{s}"]["lo"][j90] for s in PATCH])
        hi = np.array([m[f"cov_patch{s}"]["hi"][j90] for s in PATCH])
        ax[2].errorbar(PATCH, e, yerr=[e - lo, hi - e], marker="o", capsize=3, label=t, color=COLORS.get(t))
    for a_ in ax[:2]:
        a_.axhline(0, color="k", lw=0.8)
        a_.set_xlabel("nominal level")
        a_.set_ylabel("coverage - exact level")
    ax[2].set_xscale("log", base=2)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("patch size [px] (128 = whole image)")
    ax[2].set_ylabel("90% coverage - exact")
    ax[2].set_title("patch-mean coverage (90%)")
    ax[0].legend()
    save(fig, f"coverage{suffix}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for t in tags:
        m = res["models"][t]
        for a_, key, ttl in [(ax[0], "pow_ratio_sample_over_truth", "per-sample / truth"),
                             (ax[1], "pow_ratio_mean_over_truth", "posterior mean / truth"),
                             (ax[2], "bandpower_cov90_minus_exact", "band-power 90% coverage - exact")]:
            e, lo, hi = (np.array(m[key][q]) for q in ("est", "lo", "hi"))
            a_.plot(K, e, marker="o", label=t, color=COLORS.get(t))
            a_.fill_between(K, lo, hi, alpha=0.25, color=COLORS.get(t))
            a_.set_title(ttl)
            a_.set_xscale("log")
            a_.set_xlabel("|k| [cycles / 128 px]")
    ax[0].axhline(1, color="k", lw=0.8)
    ax[1].axhline(1, color="k", lw=0.8)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[0].legend()
    save(fig, f"power{suffix}")

    fig, ax = plt.subplots(1, len(tags), figsize=(4 * len(tags), 3.2))
    for a_, t in zip(np.atleast_1d(ax), tags):
        h = S[t]["rank_pix_all"][:n].sum(0)
        b = h.reshape(-1)[:N + 1]
        nb = 16
        edges = np.linspace(0, N + 1, nb + 1).round().astype(int)
        dens = np.array([b[edges[j]:edges[j + 1]].sum() for j in range(nb)]) / b.sum() / (np.diff(edges) / (N + 1))
        a_.bar(np.arange(nb), dens, color=COLORS.get(t))
        a_.axhline(1, color="k")
        a_.set_title(f"{t}: per-pixel rank histogram")
    save(fig, f"rankhist{suffix}")

    figs = {t: np.load(f"{OUT}/figs_{t}.npy", allow_pickle=True).item() for t in tags}
    for i in sorted(figs[tags[0]].keys()):
        f0 = figs[tags[0]][i]
        fig, ax = plt.subplots(len(tags), 6, figsize=(18, 3 * len(tags)))
        vmax = f0["x"].max()
        for r, t in enumerate(tags):
            f = figs[t][i]
            panels = [(f["x"], "truth"), (f["dirty"], "dirty"), (f["mean"], f"{t} mean"), (f["std"], f"{t} std"),
                      (f["samples"][0], f"{t} sample 1"), (f["samples"][1], f"{t} sample 2")]
            for c, (im, ttl) in enumerate(panels):
                kw = {} if c in (1, 3) else dict(vmin=0, vmax=vmax)
                ax[r, c].imshow(im, origin="lower", cmap="inferno", **kw)
                ax[r, c].set_title(ttl)
                ax[r, c].axis("off")
        save(fig, f"examples_img{i}{suffix}")


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_{name}.{ext}", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
