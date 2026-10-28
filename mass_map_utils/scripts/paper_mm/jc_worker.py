"""Worker for mm_jointcov.py's process pool: one jointcov.pits_case call (shared definitions, unchanged)."""
import sys

import numpy as np

sys.path.insert(0, "/home/jjwhit/crps_review/paper/common")
import jointcov  # noqa: E402


def pits_job(x, y, mask, ks, seed):
    return jointcov.pits_case(x, y, mask, ks, np.random.default_rng(seed))
