"""Marginal + joint coverage (BRIEF_jointcov.md) for the mass-mapping models, with the shared ~/crps_review/paper/common/jointcov.py.

Same cases as the joint tests: first 300 sorted test maps, N = 256 samples, observation noise seed 1e6 + i,
latent seed torch.manual_seed(i) (stage-A TestSet / sample). Image variable: kappa; region mask: the COSMOS survey mask.
Samples are generated on the fly and never stored. PITs are computed in a process pool (run_parallel). ks = 1, 2, ..., 128 (<= side/2) and None (whole footprint).
Each method is written to EVAL/../jointcov_parts/<key>.json and merged into mm/jointcov.json (schema of the brief).

Run through mm_run.py (resolves the model keys):  python mm_run.py stageA $PWD/mm_jointcov.py --models F100 MS ...
"""
import argparse
import collections
import glob
import multiprocessing as mp
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "/home/jjwhit/crps_review/paper/common")
import jointcov  # noqa: E402
from common import MODELS, TestSet, load_cfg, load_model, sample  # noqa: E402  (patched by mm_run.py)

ROOT = "/share/gpu0/jjwhit/crps_review/paper/mm/"
PARTS = ROOT + "jointcov_parts/"
# paper-mm model key -> jointcov method key (brief's names; seeds as <method>_s<n>; R_s<n> = R retrained on the paper branch)
JKEY = {"R": "R", "F100": "F", "MS": "MS", "MSS": "MSsig", "FNA": "Fnoadv", "MSNA": "MSnoadv"}
JKEY.update({f"ES{s}": "ES" if s == 0 else f"ES_s{s}" for s in range(3)})
JKEY.update({f"F{s}": f"F_s{s}" for s in (1, 2)})
JKEY.update({f"MS{s}": f"MS_s{s}" for s in (1, 2)})
JKEY.update({f"MSS{s}": f"MSsig_s{s}" for s in (1, 2)})
JKEY.update({f"R{s}": f"R_s{s}" for s in range(3)})
LABEL = {"R": "rcGAN (original)", "F": "fair CRPS", "MS": "multiscale fair CRPS", "MSsig": "MS, 1/σ level weights",
         "Fnoadv": "fair CRPS, no critic", "MSnoadv": "MS, no critic", "ES": "fair energy score"}
KS = [1, 2, 4, 8, 16, 32, 64, 128, None]


def label(j):
    base, _, s = j.partition("_s")
    if base == "R" and s:
        return f"rcGAN (paper branch), seed {s}"
    return LABEL[base] + (f", seed {s}" if s else "")


def merge(N, n_cases):
    methods = {}
    for f in sorted(glob.glob(PARTS + "*.json")):
        d = json.load(open(f))
        methods[d["key"]] = d["result"]
    notes = ("WL mass mapping (mmGAN). Cases: first {n} sorted kappa_test maps, N = {N} samples each, noise seed 1e6 + i, "
             "latent seed i (same as the joint tests). Image: kappa (300x300), regions restricted to the COSMOS survey mask "
             "(tiles kept if >= 50% unmasked, jointcov default). ks = 1..128 and full footprint. "
             "Checkpoints: best val RMSE over epochs 80-99 for every run. R = original rcGAN run (older code); R_s<n> = rcGAN "
             "loss retrained on the paper branch; F = F100 (= F seed 0); ES = fair energy score + critic, seed 0. "
             "Samples drawn on the GPU type of the job (CUDA latent RNG is hardware dependent; truths and noise identical).")
    out = {"domain": "mm", "image_side": 300, "N": N, "n_cases": n_cases, "mask": "COSMOS survey mask (True = observed)",
           "labels": {k: label(k) for k in methods}, "methods": methods, "notes": notes.format(n=n_cases, N=N)}
    tmp = ROOT + "jointcov.json.tmp"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, ROOT + "jointcov.json")


def run_parallel(sample_fn, cases, mask, ks, seed=0, n_jobs=16):
    """jointcov.run(..., null_check=True) with the per-case PITs computed in a process pool: the same jointcov.pits_case and
    Accumulator; only the randomisation stream differs (one rng per case and role, seeded by (seed, case, role))
    instead of one sequential stream. GPU sampling stays in this process; at most n_jobs + 2 cases are in flight."""
    sys.path.insert(0, "/home/jjwhit/.herdr/worktrees/rcGAN/crps-paper-mm/mass_map_utils/scripts/paper_mm")
    from jc_worker import pits_job

    # one BLAS/OpenMP thread per worker (read by the spawned children at import): unset, each of the n_jobs workers
    # starts one thread per core and the node is oversubscribed (it halved this job's and a co-located training's speed)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    acc, nul = jointcov.Accumulator(), jointcov.Accumulator()
    pending = collections.deque()
    with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn")) as ex:
        for c in cases:
            x, y = sample_fn(c)
            pending.append((ex.submit(pits_job, x, y, mask, ks, [seed, c, 0]),
                            ex.submit(pits_job, x[1:], x[0], mask, ks, [seed, c, 1])))
            while len(pending) > n_jobs + 2:
                a, b = pending.popleft()
                acc.add(a.result())
                nul.add(b.result())
        while pending:
            a, b = pending.popleft()
            acc.add(a.result())
            nul.add(b.result())
    return {"model": acc.summary(), "null": nul.summary()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--n-maps", type=int, default=300)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--part-suffix", default="", help="e.g. _test for a short timing run (not merged)")
    args = p.parse_args()

    cfg = load_cfg()
    ts = TestSet(cfg)
    mask = ts.mask
    n_maps = min(args.n_maps, len(ts))
    os.makedirs(PARTS, exist_ok=True)
    for k in args.models:
        j = JKEY[k]
        model = load_model(MODELS[k])
        t0 = time.time()

        def sample_fn(i):
            y, xn = ts.get(i)
            S = sample(model, y, args.N, chunk=args.chunk, seed=i).float().cpu().numpy()
            x = (xn[0].numpy() * cfg.kappa_std + cfg.kappa_mean).astype(np.float32)
            if i % 25 == 0 or i < 3:
                print(f"[{k}] map {i + 1}/{n_maps}  {time.time() - t0:.0f}s", flush=True)
            return S, x

        res = run_parallel(sample_fn, range(n_maps), mask, KS, seed=0, n_jobs=args.n_jobs)
        part = {"key": j, "model": k, "exp": MODELS[k], "ckpt": "checkpoint_best.ckpt", "n_maps": n_maps, "N": args.N,
                "seconds": time.time() - t0, "result": res}
        if args.part_suffix:
            json.dump(part, open(PARTS + f"{j}{args.part_suffix}.json.test", "w"))
        else:
            json.dump(part, open(PARTS + f"{j}.json", "w"))
            merge(args.N, n_maps)
        c = res["model"]["stats"]
        print(f"[{k}] done {time.time() - t0:.0f}s  pixel cov90-90 {100 * (c['pixel']['1']['cov'][17] - 0.9):+.1f} pp  "
              f"null pixel {100 * (res['null']['stats']['pixel']['1']['cov'][17] - 0.9):+.1f}", flush=True)
        del model


if __name__ == "__main__":  # spawned PIT workers import this file as __mp_main__ and must not re-run it
    main()
