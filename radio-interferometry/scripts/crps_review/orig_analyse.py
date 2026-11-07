"""CPU analysis of orig_eval.py outputs for the original RIGAN models: bootstrap 95% CIs over test images
(2000 resamples, seed 0, same indices for both models), results.json and figures.

Coverage convention (as reports/06_radio.md): central order-statistic interval [s_(l), s_(u)] of N samples, reported as
empirical - exact coverage (u - l)/(N + 1). Gaussian-interval coverage uses mean +- z sd of the same N samples.

Usage: python orig_analyse.py [unet gunet] [--base DIR]
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LEVELS = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
MASKS = ["all", "src", "bg", "bright"]
NB = 2000
LABEL = {"unet": "RIGAN U-Net", "gunet": "RIGAN GU-Net"}
COLORS = {"unet": "#1f77b4", "gunet": "#d62728"}


def interval(N, a):
    l = max(int(np.floor((1 - a) / 2 * (N + 1))), 1)
    u = N + 1 - l
    return l, u, (u - l) / (N + 1)


def cov_hist(h, N):
    """h (..., N+1) rank counts -> empirical - exact coverage per level."""
    tot = np.maximum(h.sum(-1), 1)
    return np.stack([h[..., l:u].sum(-1) / tot - ex for l, u, ex in (interval(N, a) for a in LEVELS)], -1)


def gcov_hist(h):
    """h (..., B) histogram of Gaussian PIT on [0, 1] -> coverage of mean +- z sd intervals - nominal."""
    B = h.shape[-1]
    c = (np.arange(B) + 0.5) / B
    tot = np.maximum(h.sum(-1), 1)
    return np.stack([h[..., np.abs(c - 0.5) <= a / 2].sum(-1) / tot - a for a in LEVELS], -1)


def calib_err(h, nbins=10):
    N1 = h.shape[-1]
    e = np.linspace(0, N1, nbins + 1).round().astype(int)
    b = np.stack([h[..., e[j]:e[j + 1]].sum(-1) for j in range(nbins)], -1).astype(float)
    dens = b / np.maximum(b.sum(-1, keepdims=True), 1) / (np.diff(e) / N1)
    return np.sqrt(((dens - 1) ** 2).mean(-1))


def pit_stats(h):
    """bias (mean PIT - 1/2) and dispersion 4 E|PIT - 1/2| (1 = uniform, > 1 under-dispersed samples)."""
    N1 = h.shape[-1]
    u = (np.arange(N1) + 0.5) / N1
    p = h / np.maximum(h.sum(-1, keepdims=True), 1)
    return np.stack([(p * u).sum(-1) - 0.5, 4 * (p * np.abs(u - 0.5)).sum(-1)], -1)


def hist10(h):
    N1 = h.shape[-1]
    e = np.linspace(0, N1, 11).round().astype(int)
    b = np.array([h[e[j]:e[j + 1]].sum() for j in range(10)], float)
    return b / b.sum() / (np.diff(e) / N1)


def boot(f, n, idx):
    pt = np.asarray(f(np.arange(n)))
    bs = np.stack([np.asarray(f(i)) for i in idx])
    return pt, np.quantile(bs, 0.025, 0), np.quantile(bs, 0.975, 0)


def summarise(s, idx):
    n = len(s["img"])
    N = int(s["n_samp"])
    R = {"n_images": n, "n_samp": N}
    pm = lambda k: boot(lambda i: s[k][i].mean(), n, idx)  # noqa: E731
    for k in ["psnr", "ssim", "psnr32", "ssim32", "snr32", "snrN", "corr_std_err32", "corr_std_errN"]:
        R[k] = [float(v) for v in pm(k)]
    # spread-skill: sqrt( mean var * (N+1)/N / mean mse ), 1 = calibrated spread
    for sfx in ["", "_src"]:
        R["spread_skill" + sfx] = [float(v) for v in boot(
            lambda i: np.sqrt(s["var" + sfx][i].mean() * (N + 1) / N / s["mse" + sfx][i].mean()), n, idx)]
    for m in MASKS:
        h = s[f"rank_{m}"]
        R[f"cov_pix_{m}"] = [v.tolist() for v in boot(lambda i: cov_hist(h[i].sum(0), N), n, idx)]
        R[f"calerr_pix_{m}"] = [float(v) for v in boot(lambda i: calib_err(h[i].sum(0)), n, idx)]
        R[f"pit_bias_disp_{m}"] = [v.tolist() for v in boot(lambda i: pit_stats(h[i].sum(0)), n, idx)]
        R[f"hist10_{m}"] = hist10(h.sum(0)).tolist()
        g = s[f"gpit_{m}"]
        R[f"gcov_pix_{m}"] = [v.tolist() for v in boot(lambda i: gcov_hist(g[i].sum(0)), n, idx)]
        R[f"gcalerr_pix_{m}"] = [float(v) for v in boot(lambda i: calib_err(g[i].sum(0)), n, idx)]
        sk, ku, z = s[f"skew_{m}"].sum(0), s[f"kurt_{m}"].sum(0), s[f"z_{m}"].sum(0)
        se, ke, ze = s["skew_edges"], s["kurt_edges"], s["z_edges"]
        sc, kc, zc = (se[1:] + se[:-1]) / 2, (ke[1:] + ke[:-1]) / 2, (ze[1:] + ze[:-1]) / 2

        def q(c, w, p):
            cw = np.cumsum(w) / w.sum()
            return float(c[np.searchsorted(cw, p)])
        R[f"shape_{m}"] = {
            "skew_q10_50_90": [q(sc, sk, p) for p in (0.1, 0.5, 0.9)],
            "frac_abs_skew_gt_0.5": float(sk[np.abs(sc) > 0.5].sum() / sk.sum()),
            "frac_abs_skew_gt_1": float(sk[np.abs(sc) > 1].sum() / sk.sum()),
            "exkurt_q10_50_90": [q(kc, ku, p) for p in (0.1, 0.5, 0.9)],
            "frac_exkurt_gt_1": float(ku[kc > 1].sum() / ku.sum()),
            "frac_abs_z_gt_3": float(z[np.abs(zc) > 3].sum() / z.sum()),  # Gaussian: 0.0027
            "frac_abs_z_gt_5": float(z[np.abs(zc) > 5].sum() / z.sum()),
        }
    for k in [k for k in s["patch"] if k != 360]:
        h = s[f"prank_{k}"]
        R[f"cov_patch_{k}"] = [v.tolist() for v in boot(lambda i: cov_hist(h[i].sum(0), N), n, idx)]
    hi = np.zeros((n, N + 1))
    hi[np.arange(n), s["rank_img"]] = 1
    R["cov_patch_360"] = [v.tolist() for v in boot(lambda i: cov_hist(hi[i].sum(0), N), n, idx)]
    R["hist10_img"] = hist10(hi.sum(0)).tolist()
    R["neg_frac_src"], R["neg_frac_bg"] = float(s["neg_frac_src"].mean()), float(s["neg_frac_bg"].mean())
    R["truth_frac_below_1e-3"] = float(s["zero_true_frac"].mean())
    ok = s["ss_n"] > 1000
    R["reliability"] = {"sd_bin_lo": s["sd_edges"][:-1][ok].tolist(),
                        "rms_spread": np.sqrt(s["ss_var"][ok] * (N + 1) / N / s["ss_n"][ok]).tolist(),
                        "rmse": np.sqrt(s["ss_err"][ok] / s["ss_n"][ok]).tolist(), "n_pix": s["ss_n"][ok].tolist()}
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["unet", "gunet"])
    ap.add_argument("--base", default="/share/gpu0/jjwhit/crps_review/radio_orig/")
    a = ap.parse_args()
    ev, fd = a.base + "eval/", a.base + "figs/"
    os.makedirs(fd, exist_ok=True)
    S = {m: dict(np.load(ev + f"stats_{m}.npz")) for m in a.models if os.path.exists(ev + f"stats_{m}.npz")}
    n = min(len(s["img"]) for s in S.values())
    for s in S.values():  # paired: same images for every model
        assert (s["img"][:n] == next(iter(S.values()))["img"][:n]).all()
        for k, v in list(s.items()):
            if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(s["img"]) and not k.startswith("fig"):
                s[k] = v[:n]
    idx = np.random.default_rng(0).integers(0, n, (NB, n))
    res = {"levels": LEVELS, "models": {m: summarise(s, idx) for m, s in S.items()}}
    if {"unet", "gunet"} <= set(S):
        d = S["gunet"]["psnr"] - S["unet"]["psnr"]
        res["psnr_gunet_minus_unet"] = [float(v) for v in boot(lambda i: d[i].mean(), n, idx)]
    json.dump(res, open(ev + "results.json", "w"), indent=1)
    figures(S, res, fd)
    print_tables(res)


def print_tables(res):
    L = res["levels"]
    i50, i90 = L.index(0.5), L.index(0.9)
    f = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"  # noqa: E731
    for m, R in res["models"].items():
        print(f"===== {m}: n={R['n_images']} N={R['n_samp']}")
        for k in ["psnr", "ssim", "psnr32", "ssim32", "snr32", "snrN", "corr_std_err32", "corr_std_errN", "spread_skill", "spread_skill_src"]:
            print(f"  {k:16s} {f(R[k])}")
        for msk in MASKS:
            c, g = R[f"cov_pix_{msk}"], R[f"gcov_pix_{msk}"]
            print(f"  pix {msk:6s} cov-exact 50/90: {c[0][i50]:+.3f} [{c[1][i50]:+.3f},{c[2][i50]:+.3f}] / {c[0][i90]:+.3f} [{c[1][i90]:+.3f},{c[2][i90]:+.3f}]"
                  f"  gauss 50/90: {g[0][i50]:+.3f} / {g[0][i90]:+.3f}  calerr {R[f'calerr_pix_{msk}'][0]:.3f} gcalerr {R[f'gcalerr_pix_{msk}'][0]:.3f}"
                  f"  bias/disp {R[f'pit_bias_disp_{msk}'][0][0]:+.3f}/{R[f'pit_bias_disp_{msk}'][0][1]:.3f}")
            print(f"      hist10 {np.round(R[f'hist10_{msk}'], 2).tolist()}")
            print(f"      shape {R[f'shape_{msk}']}")
        for k in [2, 4, 8, 16, 32, 64, 360]:
            c = R[f"cov_patch_{k}"]
            print(f"  patch {k:3d} cov-exact 50/90: {c[0][i50]:+.3f} [{c[1][i50]:+.3f},{c[2][i50]:+.3f}] / {c[0][i90]:+.3f} [{c[1][i90]:+.3f},{c[2][i90]:+.3f}]")
        print("  hist10 whole-image", np.round(R["hist10_img"], 2).tolist())
        print(f"  neg frac src/bg {R['neg_frac_src']:.4f}/{R['neg_frac_bg']:.4f}; truth < 1e-3: {R['truth_frac_below_1e-3']:.4f}")
    if "psnr_gunet_minus_unet" in res:
        print("psnr gunet - unet", res["psnr_gunet_minus_unet"])


def figures(S, res, fd):
    L = np.array(res["levels"])
    M = res["models"]
    # (a) per-pixel coverage curves (rank intervals solid, Gaussian intervals dashed)
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    for a_, msk in zip(ax, MASKS):
        for m, R in M.items():
            c = np.array(R[f"cov_pix_{msk}"])
            a_.plot(L, c[0], "o-", color=COLORS[m], label=f"{LABEL[m]} (sample quantiles)")
            a_.fill_between(L, c[1], c[2], color=COLORS[m], alpha=0.2)
            a_.plot(L, np.array(R[f"gcov_pix_{msk}"])[0], "s--", ms=3, color=COLORS[m], alpha=0.7, label=f"{LABEL[m]} (mean ± z·sd)")
        a_.axhline(0, color="k", lw=0.8)
        a_.set_title(f"per-pixel coverage, mask '{msk}'")
        a_.set_xlabel("nominal level")
    ax[0].set_ylabel("empirical − nominal")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); save(fig, fd, "fig_pixel_coverage")
    # (b) rank histograms
    fig, ax = plt.subplots(1, 5, figsize=(19, 3.4), sharey=True)
    for a_, msk in zip(ax, MASKS + ["img"]):
        for j, (m, R) in enumerate(M.items()):
            h = R[f"hist10_{msk}"]
            a_.bar(np.arange(10) + 0.4 * j - 0.2, h, width=0.4, color=COLORS[m], label=LABEL[m])
        a_.axhline(1, color="k", lw=0.8)
        a_.set_title("whole-image mean (total flux)" if msk == "img" else f"per-pixel rank histogram, '{msk}'", fontsize=9)
        a_.set_xlabel("PIT decile")
    ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)
    fig.tight_layout(); save(fig, fd, "fig_rank_hist")
    # (c) patch-mean coverage vs scale
    ks = [1, 2, 4, 8, 16, 32, 64, 360]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a_, lev in zip(ax, [0.5, 0.9]):
        i = list(L).index(lev)
        for m, R in M.items():
            c = np.array([R["cov_pix_all"] if k == 1 else R[f"cov_patch_{k}"] for k in ks])[:, :, i]
            a_.errorbar(range(len(ks)), c[:, 0], yerr=[c[:, 0] - c[:, 1], c[:, 2] - c[:, 0]], fmt="o-", color=COLORS[m], label=LABEL[m], capsize=2)
        a_.axhline(0, color="k", lw=0.8)
        a_.set_xticks(range(len(ks))); a_.set_xticklabels([str(k) if k < 360 else "image" for k in ks])
        a_.set_xlabel("patch size [px]"); a_.set_title(f"patch-mean {int(lev * 100)}% coverage")
    ax[0].set_ylabel("empirical − exact"); ax[0].legend()
    fig.tight_layout(); save(fig, fd, "fig_patch_coverage")
    # (d) spread-skill reliability
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m, R in M.items():
        r = R["reliability"]
        ax[0].loglog(r["rms_spread"], r["rmse"], "o-", color=COLORS[m], label=LABEL[m])
        ax[1].scatter(np.sqrt(S[m]["var"]), np.sqrt(S[m]["mse"]), s=3, color=COLORS[m], alpha=0.4, label=LABEL[m])
    for a_ in ax:
        lo, hi = a_.get_xlim()
        a_.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax[0].set_xlabel("RMS spread (pixels binned by posterior sd)"); ax[0].set_ylabel("RMSE of posterior mean"); ax[0].legend()
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("per-image RMS spread"); ax[1].set_ylabel("per-image RMSE")
    fig.tight_layout(); save(fig, fd, "fig_spread_skill")
    # (e) per-pixel posterior shape
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for m, s in S.items():
        for msk, ls in [("src", "-"), ("bg", ":")]:
            for a_, k, e in [(ax[0], "skew", "skew_edges"), (ax[1], "kurt", "kurt_edges"), (ax[2], "z", "z_edges")]:
                h = s[f"{k}_{msk}"].sum(0).astype(float)
                c = (s[e][1:] + s[e][:-1]) / 2
                a_.plot(c, h / h.sum() / np.diff(s[e]), ls, color=COLORS[m], label=f"{LABEL[m]}, {msk}")
    zc = np.linspace(-8, 8, 400)
    ax[2].plot(zc, np.exp(-zc ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=0.8, label="N(0,1)")
    ax[0].set_xlabel("per-pixel sample skewness"); ax[1].set_xlabel("per-pixel sample excess kurtosis")
    ax[2].set_xlabel("(truth − mean) / sd"); ax[2].set_yscale("log"); ax[2].set_ylim(1e-6, 1)
    ax[0].legend(fontsize=7); ax[2].legend(fontsize=7)
    fig.tight_layout(); save(fig, fd, "fig_posterior_shape")
    # (f) examples
    ims = sorted({int(k[3:].split("_")[0]) for k in next(iter(S.values())) if k.startswith("fig")})
    if ims:
        fig, ax = plt.subplots(len(ims), 2 + 3 * len(S), figsize=(2.6 * (2 + 3 * len(S)), 2.6 * len(ims)), squeeze=False)
        for r, i in enumerate(ims):
            s0 = next(iter(S.values()))
            ax[r, 0].imshow(s0[f"fig{i}_x"], origin="lower", cmap="afmhot", vmin=0, vmax=1); ax[r, 0].set_title("truth", fontsize=8)
            ax[r, 1].imshow(s0[f"fig{i}_dirty"], origin="lower", cmap="afmhot"); ax[r, 1].set_title("dirty", fontsize=8)
            for j, (m, s) in enumerate(S.items()):
                ax[r, 2 + 3 * j].imshow(s[f"fig{i}_mean"], origin="lower", cmap="afmhot", vmin=0, vmax=1)
                ax[r, 2 + 3 * j].set_title(f"{LABEL[m]} mean", fontsize=8)
                ax[r, 3 + 3 * j].imshow(s[f"fig{i}_samples"][0], origin="lower", cmap="afmhot", vmin=0, vmax=1)
                ax[r, 3 + 3 * j].set_title("one sample", fontsize=8)
                ax[r, 4 + 3 * j].imshow(np.log10(s[f"fig{i}_sd"] + 1e-6), origin="lower", cmap="viridis")
                ax[r, 4 + 3 * j].set_title("log10 sd", fontsize=8)
        for a_ in ax.flat:
            a_.axis("off")
        fig.tight_layout(); save(fig, fd, "fig_examples")


def save(fig, fd, name):
    for ext in ("png", "pdf"):
        fig.savefig(fd + f"{name}.{ext}", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
