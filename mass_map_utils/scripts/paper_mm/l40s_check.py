"""Compare the joint-test outputs of an L40S run (CPU FFTs) with the original A100/V100 run, map by map."""
import glob, os, sys
import numpy as np
new, old = sys.argv[1], sys.argv[2]
worst = 0.0
for f in sorted(glob.glob(os.path.join(new, "*.npy"))):
    a = np.load(f); b = np.load(os.path.join(old, os.path.basename(f)))[: a.shape[0]]
    if not np.issubdtype(a.dtype, np.floating):
        print(f"{os.path.basename(f):24s} equal={np.array_equal(a, b)}"); continue
    rel = np.nanmax(np.abs(a - b)) / (np.nanmax(np.abs(b)) + 1e-30)
    worst = max(worst, rel)
    print(f"{os.path.basename(f):24s} shape {a.shape}  max|diff|/max|old| = {rel:.2e}")
print("worst", worst)
