"""How non-Gaussian are the WL posteriors? (reports/10_mm_cfid_vs_rmse.md). Run via mm_run.py (stageA mode).

First n test maps (fixed noise and latent seeds as stage A), N samples per map. Inside the COSMOS footprint:
  * per-pixel sample skewness and excess kurtosis, with a finite-N Gaussian null (same N, same number of pixels);
  * the same for patch means (16 / 64 px tiles >= 50% unmasked) and the footprint mean;
  * coverage of the truth by central intervals from the empirical sample quantiles vs from a Gaussian with the sample mean
    and sd (what pure variance matching can represent), at 50 / 90 / 95%;
  * skewness / excess kurtosis of the truth kappa over the footprint (prior non-Gaussianity), per map.
Output: eval/gauss/gauss.json.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from scipy import stats

from common import MODELS, TestSet, load_cfg, load_model, sample  # noqa: E402  (patched by mm_run.py)

OUT = "/share/gpu0/jjwhit/crps_review/paper/mm/eval/gauss/"
LEVELS = [0.5, 0.9, 0.95]


def moments(s):  # s: (N, M) samples x variables -> skew, excess kurtosis per variable
    return stats.skew(s, axis=0), stats.kurtosis(s, axis=0)


def cover(s, t):  # s: (N, M), t: (M,) -> {level: (empirical-quantile coverage, Gaussian coverage)}
    out = {}
    mu, sd = s.mean(0), s.std(0, ddof=1)
    for lv in LEVELS:
        lo, hi = np.quantile(s, [(1 - lv) / 2, (1 + lv) / 2], axis=0)
        z = stats.norm.ppf((1 + lv) / 2)
        out[lv] = (float(np.mean((t >= lo) & (t <= hi))), float(np.mean(np.abs(t - mu) <= z * sd)))
    return out


def tiles(a, mask, k):  # a: (..., H, W) -> (..., T) means over k x k tiles with >= 50% footprint
    H, W = mask.shape
    idx = [(i, j) for i in range(0, H - k + 1, k) for j in range(0, W - k + 1, k) if mask[i:i + k, j:j + k].mean() >= 0.5]
    return np.stack([a[..., i:i + k, j:j + k][..., mask[i:i + k, j:j + k]].mean(-1) for i, j in idx], -1)


def summ(v):
    q = np.quantile(v, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {"mean": float(np.mean(v)), "q01_05_25_50_75_95_99": [float(x) for x in q]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["R", "R0", "F100", "F1"])
    p.add_argument("--n-maps", type=int, default=100)
    p.add_argument("--N", type=int, default=256)
    a = p.parse_args()
    cfg = load_cfg()
    ts = TestSet(cfg)
    mask = ts.mask
    os.makedirs(OUT, exist_ok=True)
    res = {"n_maps": a.n_maps, "N": a.N, "levels": LEVELS, "models": {}}

    # finite-N Gaussian null for the moment estimators
    rng = np.random.default_rng(0)
    g = rng.standard_normal((a.N, 200_000))
    gs, gk = moments(g)
    res["null_gauss"] = {"skew": summ(gs), "exkurt": summ(gk)}
    sk_lim, ku_lim = np.quantile(np.abs(gs), 0.99), np.quantile(gk, [0.005, 0.995])

    # prior non-Gaussianity: truth kappa over the footprint, per map
    tsk, tku = [], []
    for i in range(a.n_maps):
        x = ts.get(i)[1][0].numpy() * cfg.kappa_std + cfg.kappa_mean
        tsk.append(stats.skew(x[mask])), tku.append(stats.kurtosis(x[mask]))
    res["truth_field"] = {"skew": summ(np.array(tsk)), "exkurt": summ(np.array(tku))}

    for k in a.models:
        model = load_model(MODELS[k])
        t0 = time.time()
        acc = {"pix_skew": [], "pix_kurt": [], "pix_cov": [], "pix_mu": []}
        for sc in (16, 64):
            acc[f"t{sc}_skew"], acc[f"t{sc}_kurt"], acc[f"t{sc}_cov"] = [], [], []
        acc["fp_mean"], acc["fp_truth"] = [], []
        for i in range(a.n_maps):
            y, xn = ts.get(i)
            S = sample(model, y, a.N, chunk=64, seed=i).float().cpu().numpy()  # (N, H, W) kappa
            x = xn[0].numpy() * cfg.kappa_std + cfg.kappa_mean
            s, t = S[:, mask], x[mask]
            sk, ku = moments(s)
            acc["pix_skew"].append(sk), acc["pix_kurt"].append(ku), acc["pix_mu"].append(s.mean(0))
            acc["pix_cov"].append(cover(s, t))
            for sc in (16, 64):
                st, tt = tiles(S, mask, sc), tiles(x, mask, sc)
                sk, ku = moments(st)
                acc[f"t{sc}_skew"].append(sk), acc[f"t{sc}_kurt"].append(ku), acc[f"t{sc}_cov"].append(cover(st, tt))
            acc["fp_mean"].append(s.mean(1)), acc["fp_truth"].append(t.mean())
            if i % 20 == 0:
                print(f"[{k}] map {i + 1}/{a.n_maps} {time.time() - t0:.0f}s", flush=True)
        r = {}
        for key in ["pix", "t16", "t64"]:
            sk, ku = np.concatenate(acc[f"{key}_skew"]), np.concatenate(acc[f"{key}_kurt"])
            r[key] = {"skew": summ(sk), "exkurt": summ(ku), "n": int(sk.size),
                      "frac_skew_beyond_null99": float(np.mean(np.abs(sk) > sk_lim)),
                      "frac_kurt_beyond_null99": float(np.mean((ku < ku_lim[0]) | (ku > ku_lim[1]))),
                      "cov_empirical_vs_gauss": {str(lv): [float(np.mean([c[lv][0] for c in acc[f"{key}_cov"]])),
                                                           float(np.mean([c[lv][1] for c in acc[f"{key}_cov"]]))]
                                                 for lv in LEVELS}}
        # pixels with the largest posterior means (peaks / halos): top 1% of sample-mean kappa per map
        sk, mu = np.concatenate(acc["pix_skew"]), np.concatenate(acc["pix_mu"])
        top = mu >= np.quantile(mu, 0.99)
        r["pix_top1pct_mean"] = {"skew": summ(sk[top]), "exkurt": summ(np.concatenate(acc["pix_kurt"])[top])}
        fm, ft = np.stack(acc["fp_mean"]), np.array(acc["fp_truth"])  # (maps, N)
        fsk, fku = moments(fm.T)
        r["footprint_mean"] = {"skew": summ(fsk), "exkurt": summ(fku), "cov_empirical_vs_gauss": {
            str(lv): list(cover(fm.T, ft)[lv]) for lv in LEVELS}}
        res["models"][k] = r
        print(f"[{k}] done {time.time() - t0:.0f}s  pix skew median {r['pix']['skew']['q01_05_25_50_75_95_99'][3]:+.3f} "
              f"exkurt median {r['pix']['exkurt']['q01_05_25_50_75_95_99'][3]:+.3f}", flush=True)
        json.dump(res, open(OUT + "gauss.json", "w"), indent=1)
        del model
        torch.cuda.empty_cache()


main()
