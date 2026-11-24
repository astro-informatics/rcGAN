"""Radio (simulated MeerKAT, sparse positive sky): where does rcGAN (R) lose to the fair CRPS (F)?

1-D theory (candidates/shape_effect.py) predicts the beta-tuned rcGAN minimiser loses most on zero-inflated / strongly
skewed per-pixel posteriors (CRPS excess 10-110%), and puts 35-49% of its mass below the support (x < 0). This script
checks whether the existing epoch-69 radio models (paper-radio agent, 70 epochs) show that, stratifying pixels by
- the dirty image relative to its own maximum, d / max(d) (a function of y only, so the stratified score stays proper; the
  thermal noise is only 2.4e-3 of the sky peak, so uncertainty comes from the missing uv cells, not from noise), and
- the truth (diagnostic only: conditioning on the outcome).
Per model and stratum: per-pixel fair CRPS on raw samples and on positivity-clipped samples, the fraction of sample mass
below 0, the sample skewness (median), and clipped randomised-PIT 50%/90% coverage error. Paired R - F CRPS differences with
a bootstrap over test images. Same test images and latent seeds (1000 + i) as eval_sample.py.
-> /share/gpu0/jjwhit/crps_review/candidates/radio_strata.json
"""
import argparse
import json
import sys

import numpy as np
import torch

ROOT = "/home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-radio"
sys.path.insert(0, ROOT)
from models.lightning.riGAN import riGAN  # noqa: E402

RAD = "/share/gpu0/jjwhit/crps_review/paper/radio/"
DST = "/share/gpu0/jjwhit/crps_review/candidates/radio_strata.json"
SNR_EDGES = [-np.inf, 0.02, 0.05, 0.1, 0.2, 0.4, np.inf]  # d / max(d)
TRU_EDGES = [-1, 0, 0.01, 0.05, 0.2, np.inf]  # (-1, 0] = exactly 0

ap = argparse.ArgumentParser()
ap.add_argument("--names", nargs="+", default=["R", "F", "MSsigma"])
ap.add_argument("--n-test", type=int, default=1000)
ap.add_argument("--n-samp", type=int, default=128)
a = ap.parse_args()
dev = "cuda"
N = a.n_samp

tr = RAD + "data/train/"
mx, sx, my, sy, mu, su = [float(np.load(tr + f"{k}.npy")) for k in ["mean_x", "std_x", "mean_y", "std_y", "mean_uv", "std_uv"]]
X = np.load(RAD + "data/test/x.npy", mmap_mode="r")
Y = np.load(RAD + "data/test/y.npy", mmap_mode="r")
UV = np.load(RAD + "data/test/uv.npy", mmap_mode="r")
gen = torch.Generator(device=dev).manual_seed(0)
KEYS = ["n", "crps", "crps_clip", "neg", "cov50", "cov90"]


def fair_crps_pix(S, t):
    """S (N, P) samples, t (P,) -> (P,) fair CRPS."""
    s, _ = S.sort(0)
    i = torch.arange(1, N + 1, device=S.device, dtype=S.dtype)[:, None]
    return (S - t).abs().mean(0) - ((2 * i - N - 1) * s).sum(0) / (N * (N - 1))


def clip_pit(S, t):
    Sc = S.clamp_min(0)
    below = (Sc < t).sum(0).double()
    eq = (Sc == t).sum(0).double()
    u = torch.rand(t.shape, device=dev, generator=gen, dtype=torch.float64)
    return (below + u * (eq + 1)) / (N + 1)


res = {}
for name in a.names:
    model = riGAN.load_from_checkpoint(RAD + f"models/radio_sim_{name}/checkpoint-epoch=69.ckpt", map_location="cpu").to(dev).eval()
    print(name, "reg", model.reg, "std_mult", float(model.std_mult), flush=True)
    # per-image, per-stratum sums: [n_img, n_strata, len(KEYS)]
    acc = {"snr": np.zeros((a.n_test, len(SNR_EDGES) - 1, len(KEYS))), "truth": np.zeros((a.n_test, len(TRU_EDGES) - 1, len(KEYS)))}
    skew = {"snr": [[] for _ in SNR_EDGES[1:]], "truth": [[] for _ in TRU_EDGES[1:]]}
    with torch.no_grad():
        for i in range(a.n_test):
            x = torch.from_numpy(np.array(X[i])).float().to(dev).reshape(-1).double()
            d = torch.from_numpy(np.array(Y[i])).float().to(dev)
            y = torch.stack([(d - my) / sy, (torch.from_numpy(np.array(UV[i])).float().to(dev) - mu) / su])[None]
            torch.manual_seed(1000 + i)
            S = torch.cat([model(y.repeat(min(32, N - j), 1, 1, 1)) for j in range(0, N, 32)])[:, 0]
            S = (S * sx + mx).reshape(N, -1).double()
            c = fair_crps_pix(S, x)
            cc = fair_crps_pix(S.clamp_min(0), x)
            neg = (S < 0).double().mean(0)
            u = clip_pit(S, x)
            c50 = ((u - 0.5).abs() <= 0.25).double()
            c90 = ((u - 0.5).abs() <= 0.45).double()
            m = S.mean(0)
            sk = ((S - m) ** 3).mean(0) / S.var(0, unbiased=False).clamp_min(1e-30) ** 1.5
            snr = d.reshape(-1).double() / d.max()
            for key, feat, edges in [("snr", snr, SNR_EDGES), ("truth", x, TRU_EDGES)]:
                b = torch.bucketize(feat, torch.tensor(edges[1:-1], device=dev, dtype=feat.dtype), right=False)
                for k in range(len(edges) - 1):
                    sel = b == k
                    nk = int(sel.sum())
                    if nk == 0:
                        continue
                    acc[key][i, k] = [nk, float(c[sel].sum()), float(cc[sel].sum()), float(neg[sel].sum()),
                                      float(c50[sel].sum()), float(c90[sel].sum())]
                    if i % 10 == 0:  # skewness subsample
                        skew[key][k].append(sk[sel][:: max(1, nk // 200)].cpu().numpy())
            if i % 200 == 0:
                print(name, i, flush=True)
    out = {}
    for key, edges in [("snr", SNR_EDGES), ("truth", TRU_EDGES)]:
        A = acc[key]
        tot = A.sum(0)
        n = tot[:, 0]
        out[key] = dict(edges=[str(e) for e in edges], n_pix=n.tolist(),
                        crps=(tot[:, 1] / n).tolist(), crps_clip=(tot[:, 2] / n).tolist(), frac_neg=(tot[:, 3] / n).tolist(),
                        cov50_err=(tot[:, 4] / n - 0.5).tolist(), cov90_err=(tot[:, 5] / n - 0.9).tolist(),
                        skew_median=[float(np.median(np.concatenate(s))) if s else None for s in skew[key]],
                        skew_q90=[float(np.quantile(np.concatenate(s), 0.9)) if s else None for s in skew[key]])
        np.save(DST.replace(".json", f"_{name}_{key}.npy"), A)
    allA = acc["snr"].sum(1).sum(0)
    out["all"] = dict(crps=allA[1] / allA[0], crps_clip=allA[2] / allA[0], frac_neg=allA[3] / allA[0])
    res[name] = out
    print(name, json.dumps(out), flush=True)
    del model
    torch.cuda.empty_cache()


def boot_diff(key, col, ref="F", B=2000):
    rng = np.random.default_rng(0)
    out = {}
    for name in res:
        if name == ref:
            continue
        A, Bm = np.load(DST.replace(".json", f"_{name}_{key}.npy")), np.load(DST.replace(".json", f"_{ref}_{key}.npy"))
        rel = []
        for _ in range(B):
            idx = rng.integers(0, len(A), len(A))
            sa, sb = A[idx].sum(0), Bm[idx].sum(0)
            rel.append(sa[:, col] / sb[:, col] - 1)
        rel = np.array(rel)
        pt = A.sum(0)[:, col] / Bm.sum(0)[:, col] - 1
        out[name] = dict(rel_excess=pt.tolist(), ci95=np.nanquantile(rel, [0.025, 0.975], axis=0).T.tolist())
    return out


res["paired_vs_F"] = {f"{key}_{lab}": boot_diff(key, col) for key in ["snr", "truth"] for lab, col in [("crps", 1), ("crps_clip", 2)]}
json.dump(res, open(DST, "w"), indent=1, default=float)
print(json.dumps(res["paired_vs_F"], indent=1))
