"""Marginal + joint coverage (shared ~/crps_review/paper/common/jointcov.py) for the radio variants.

Same test set, samples and latent seeds as eval_sample.py (first --n-test test images, torch.manual_seed(1000 + i),
N = 128 (chunk 4 by default; eval_sample.py used 32), generator in eval mode, sky units). Raw samples (no positivity clipping). Samples are generated on the
fly and never stored. Variants: 'none' (raw samples, full 128x128 field; primary), 'src' (raw, restricted to the
y-measurable source mask of eval_sample.py, per image) and 'clip' (samples clamped at 0, full field). Each (method, variant)
result goes to jointcov_parts/<variant>/<key>.json and is merged under a file lock into jointcov.json / jointcov_srcmask.json /
jointcov_clip.json, all in the BRIEF_jointcov schema.

Usage: python jointcov_radio.py --method R [--variant none|src|clip] [--n-test 1000]
"""
import argparse
import fcntl
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/jjwhit/crps_review/paper/common")
import jointcov  # noqa: E402
from eval_sample import DATA, source_mask  # noqa: E402
from models.lightning.riGAN import riGAN  # noqa: E402  (eval_sample puts the worktree on sys.path)

BASE = "/share/gpu0/jjwhit/crps_review/paper/radio/"
CKPT = {"R": "radio_sim_R", "F": "radio_sim_F", "MS": "radio_sim_MS", "MSsig": "radio_sim_MSsigma", "Fnoadv": "radio_sim_Fnoadv"}
LABELS = {"R": "rcGAN", "F": "fair CRPS", "MS": "MS fair CRPS", "MSsig": "MS fair CRPS (1/sigma_s)", "Fnoadv": "fair CRPS, no critic"}
KS = (1, 2, 4, 8, 16, 32, 64, None)
OUTS = {"none": BASE + "jointcov.json", "src": BASE + "jointcov_srcmask.json", "clip": BASE + "jointcov_clip.json"}
MASKDESC = {"none": "none (full 128x128 field)", "src": "per-image source mask (smoothed dirty > 0.1 max, dilated 2 px)",
            "clip": "none (full 128x128 field); samples positivity-clipped"}
NOTES = {
    "none": ("Simulated MeerKAT fallback data (thresholded kappaTNG sky, 128 px; 86% of truth pixels exactly 0). Test set = first n_cases "
             "of radio/data/test, fixed noise; N = 128 samples per image, latent seeds torch.manual_seed(1000+i), final checkpoint (epoch 69) "
             "of every variant, same per-image seeds as reports/06_radio.md but generated in chunks of 4 (06_radio.md: 32), so the latent draws differ from 06_radio.md (Monte Carlo-level differences only). Raw samples (no positivity clipping): with the zero atom, per-pixel "
             "PITs at background pixels depend on how the sample noise floor sits around 0, so the 'pixel' curve is the raw-rank diagnostic "
             "of 06_radio.md (cov_pix_all), not its positivity-clipped variant. Full field, no mask."),
    "src": ("As jointcov.json but restricted to the per-image y-measurable source mask (Gaussian-smoothed dirty image > 0.1 max, "
            "dilated 2 px; ~31% of pixels); tiles need >= 50% of their pixels in the mask; 'full' = all mask pixels."),
    "clip": ("As jointcov.json (full field) but with samples clamped at 0 (monotone projection onto the sky support x >= 0), the "
             "variant used for per-pixel calibration in 06_radio.md. Truth and clipped samples tie at 0 on background pixels: ties are "
             "randomised by jointcov._pit (pixel/mean/avgrank), and pixels where the whole pool is 0 have s = 0 -> inf and carry no "
             "information in maxz/depth (jointcov's convention)."),
}


def merge(mask_name, key, res, meta, out):
    path = out + os.path.basename(OUTS[mask_name])
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.load(open(path)) if os.path.exists(path) else {}
        d.update({k: v for k, v in meta.items() if k not in ("labels", "methods")})
        d.setdefault("labels", {})[key] = LABELS[key]
        d.setdefault("methods", {})[key] = res
        tmp = path + ".tmp"
        json.dump(d, open(tmp, "w"))
        os.replace(tmp, path)
        fcntl.flock(lk, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(CKPT))
    ap.add_argument("--variant", default="none", choices=["none", "src", "clip"])
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-samp", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=4)  # small chunks: ~8 processes share one 16 GB V100
    ap.add_argument("--out", default=BASE, help="output dir (a scratch dir for smoke tests)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = riGAN.load_from_checkpoint(BASE + f"models/{CKPT[a.method]}/checkpoint-epoch=69.ckpt", map_location="cpu").to(dev).eval()
    tr = DATA + "train/"
    mx, sx, my, sy, mu, su = [float(np.load(tr + f"{k}.npy")) for k in ["mean_x", "std_x", "mean_y", "std_y", "mean_uv", "std_uv"]]
    X = np.load(DATA + "test/x.npy")[: a.n_test]
    Y = np.load(DATA + "test/y.npy")[: a.n_test]
    UV = np.load(DATA + "test/uv.npy")[: a.n_test]
    N = a.n_samp
    t0 = time.time()

    def sample_fn(i):
        with torch.no_grad():
            d = torch.from_numpy(Y[i]).float().to(dev)
            y = torch.stack([(d - my) / sy, (torch.from_numpy(UV[i]).float().to(dev) - mu) / su])[None]
            torch.manual_seed(1000 + i)
            S = torch.cat([model(y.repeat(min(a.chunk, N - j), 1, 1, 1)) for j in range(0, N, a.chunk)])[:, 0]
            S = S * sx + mx
            if a.variant == "clip":
                S = S.clamp_min(0)
            S = S.cpu().numpy()
        if i % 100 == 0:
            print(a.method, i, f"{time.time() - t0:.0f}s", flush=True)
        return S, X[i]

    def src(i):
        return source_mask(torch.from_numpy(Y[i]).float()).numpy()

    meta = {"domain": "radio", "image_side": 128, "N": N, "n_cases": int(len(X))}
    for mname in [a.variant]:
        t0 = time.time()
        res = jointcov.run(sample_fn, range(len(X)), mask=src if mname == "src" else None, ks=KS, seed=0, null_check=True)
        print(a.method, mname, f"done in {time.time() - t0:.0f}s", flush=True)
        os.makedirs(a.out + f"jointcov_parts/{mname}", exist_ok=True)
        json.dump(res, open(a.out + f"jointcov_parts/{mname}/{a.method}.json", "w"))
        meta_m = dict(meta, mask=MASKDESC[mname], notes=NOTES[mname])
        merge(mname, a.method, res, meta_m, a.out)


if __name__ == "__main__":
    main()
