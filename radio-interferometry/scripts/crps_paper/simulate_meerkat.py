"""Simulated MeerKAT-like RI data set for the CRPS paper (fallback for the RIGAN TNG data, which no
longer exists on /share/gpu0; CASA/simms/wsclean are not installed either).

Pipeline per image (mirrors radio-interferometry/scripts/create_radio_data.py where possible):
  sky:   a random 128x128 crop of a kappaTNG convergence map (stand-in sky), thresholded at a global
         constant t and clipped at 0 (sparse, positive "sources" = halos), then x / max(x) as in RIGAN.
  uv:    MeerKAT 64-antenna ITRF layout (the table simms uses), Earth-rotation synthesis like the simms
         call in create_radio_data.py: random dec in [-70, 10] deg, 8 h track (HA -4..4 h), 240 s
         integrations, 10 channels of 10 MHz from f0 ~ U(800, 1500) MHz. Pixel 2 arcsec.
  vis:   gridded (cell = 1/FoV) visibilities = FFT of the sky at the cell (no degridding error) plus
         complex Gaussian noise with variance sigma_vis^2 / n_k (n_k = visibilities in cell k, Hermitian).
  image: Briggs (robust 0.5) gridded weights, dirty image and PSF normalised to PSF peak 1, as wsclean.
Output: x.npy (sky), y.npy (dirty), uv.npy (PSF), meta.npz, in data_dir/{train,val,test}.
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np

C = 299792458.0
LON = np.deg2rad(21.4430)
ARCSEC = np.pi / 180 / 3600
ITRF = "/share/gpu0/jjwhit/crps_review/paper/radio/meerkat/meerkat.itrf.txt_7eb1"
KAPPA = "/share/gpu0/jjwhit/kappa_cosmos_simulations"

N = 128
PIX = 2.0 * ARCSEC
THRESH = None  # set in main from the kappa training maps
SIGMA_IMG_NAT = None  # target rms of the natural-weighted dirty-image noise (sky peak = 1)
ROBUST = 0.5

_xyz = np.loadtxt(ITRF, usecols=(0, 1, 2))
_i, _j = np.triu_indices(len(_xyz), 1)
BASELINES = _xyz[_j] - _xyz[_i]  # (2016, 3) metres, ITRF


def uv_points(rng):
    """(u, v) in wavelengths for one random MeerKAT track, both signs (Hermitian)."""
    dec = np.deg2rad(-30 + rng.uniform(-40, 40))
    f0 = 800e6 + 700e6 * rng.uniform()
    freqs = f0 + 10e6 * np.arange(10)
    ha = np.deg2rad(15 * (np.arange(120) * 240 / 3600 - 4 + 120 / 3600))  # local HA, centred, 240 s
    H = ha - LON  # Greenwich hour angle = local HA - east longitude (ITRF x-axis at Greenwich)
    Lx, Ly, Lz = BASELINES.T
    sH, cH = np.sin(H)[:, None], np.cos(H)[:, None]
    u = sH * Lx + cH * Ly
    v = -np.sin(dec) * cH * Lx + np.sin(dec) * sH * Ly + np.cos(dec) * Lz
    u = (u.ravel()[:, None] * freqs / C).ravel()
    v = (v.ravel()[:, None] * freqs / C).ravel()
    return np.concatenate([u, -u]), np.concatenate([v, -v]), np.rad2deg(dec), f0


def grid_counts(u, v):
    du = 1.0 / (N * PIX)
    iu = np.round(u / du).astype(np.int64)
    iv = np.round(v / du).astype(np.int64)
    ok = (np.abs(iu) < N // 2) & (np.abs(iv) < N // 2)
    # numpy fft layout: index k (row = v, col = u), negative frequencies wrap
    cnt = np.bincount((iv[ok] % N) * N + (iu[ok] % N), minlength=N * N).reshape(N, N).astype(np.float64)
    return cnt, ok.mean()


def briggs(cnt):
    f2 = (5 * 10 ** (-ROBUST)) ** 2 / ((cnt ** 2).sum() / cnt.sum())
    return cnt / (1 + cnt * f2)


def sky_from_kappa(kmap, rng):
    a, b = rng.integers(0, kmap.shape[0] - N, size=2)
    x = np.clip(kmap[a:a + N, b:b + N] - THRESH, 0, None)
    return x / x.max() if x.max() > 0 else x


def observe(x, rng):
    u, v, dec, f0 = uv_points(rng)
    cnt, frac_in = grid_counts(u, v)
    nvis = cnt.sum()
    sigma_vis = SIGMA_IMG_NAT * np.sqrt(nvis)  # natural-weighting image noise = sigma_vis / sqrt(nvis)
    w = briggs(cnt)
    X = np.fft.fft2(x)
    eps = np.fft.fft2(rng.standard_normal((N, N))) / N  # Hermitian, unit complex variance
    E = np.where(cnt > 0, sigma_vis * eps / np.sqrt(np.maximum(cnt, 1)), 0)
    norm = N * N / w.sum()
    dirty = np.fft.ifft2(w * (X + E)).real * norm
    psf = np.fft.fftshift(np.fft.ifft2(w).real * norm)
    noise_only = np.fft.ifft2(w * E).real * norm
    return dirty, psf, dict(dec=dec, f0=f0, frac_in=frac_in, sampled=(cnt > 0).mean(), noise_rms=noise_only.std())


def task(args):
    path, seed = args
    rng = np.random.default_rng(seed)
    x = sky_from_kappa(np.load(path), rng)
    d, p, meta = observe(x, rng)
    return x.astype(np.float32), d.astype(np.float32), p.astype(np.float32), meta


def main():
    global THRESH, SIGMA_IMG_NAT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/share/gpu0/jjwhit/crps_review/paper/radio/data/")
    ap.add_argument("--n-train", type=int, default=8500)
    ap.add_argument("--n-val", type=int, default=499)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--thresh-pct", type=float, default=85.0)
    ap.add_argument("--sigma-img", type=float, default=2e-3)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()

    trains = sorted(glob.glob(f"{KAPPA}/kappa_train/*.npy"))
    sub = np.stack([np.load(f)[::4, ::4] for f in trains[:50]])
    THRESH = float(np.percentile(sub, a.thresh_pct))
    SIGMA_IMG_NAT = a.sigma_img
    print("threshold", THRESH, "sigma_img_nat", SIGMA_IMG_NAT, flush=True)

    splits = {
        "train": (trains[: a.n_train], 0),
        "val": (sorted(glob.glob(f"{KAPPA}/kappa_val/*.npy"))[: a.n_val], 1_000_000),
        "test": (sorted(glob.glob(f"{KAPPA}/kappa_test/*.npy"))[: a.n_test], 2_000_000),
    }
    for name, (files, s0) in splits.items():
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        with Pool(a.procs, initializer=_init, initargs=(THRESH, SIGMA_IMG_NAT)) as pool:
            res = pool.map(task, [(f, s0 + i) for i, f in enumerate(files)], chunksize=8)
        x, y, uv, meta = zip(*res)
        np.save(f"{d}/x.npy", np.stack(x))
        np.save(f"{d}/y.npy", np.stack(y))
        np.save(f"{d}/uv.npy", np.stack(uv))
        np.savez(f"{d}/meta.npz", files=np.array(files), seeds=s0 + np.arange(len(files)),
                 **{k: np.array([m[k] for m in meta]) for k in meta[0]})
        nr = np.array([m["noise_rms"] for m in meta])
        print(name, len(x), "noise rms (briggs) median", np.median(nr),
              "sampled frac", np.median([m["sampled"] for m in meta]),
              "nonzero frac", np.mean(np.stack(x) > 0), flush=True)
    np.save(os.path.join(a.out, "sim_params.npy"), dict(thresh=THRESH, sigma_img_nat=SIGMA_IMG_NAT, N=N,
            pix_arcsec=2.0, robust=ROBUST, thresh_pct=a.thresh_pct))


def _init(t, s):
    global THRESH, SIGMA_IMG_NAT
    THRESH, SIGMA_IMG_NAT = t, s


if __name__ == "__main__":
    main()
