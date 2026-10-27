"""Scan fastMRI brain multicoil dirs and record AXT2 volumes with their k-space shape.

Applies the rcGAN repo's eligibility filters (data/datasets/mri_data.py):
  train: acquisition == AXT2, kspace width >= 384, coils > 8
  val/test: acquisition == AXT2, kspace width > 384, coils > 8
"""
import json
import sys
from pathlib import Path

import h5py

ROOT = Path('/share/gpu0/tl3/fastMRI_dataset')
OUT = Path('/share/gpu0/jjwhit/crps_review/paper/mri/manifest/scan.json')

res = {}
for split in ['multicoil_train', 'multicoil_val', 'small_T2_test']:
    rows = []
    for fname in sorted((ROOT / split).iterdir()):
        if 'AXT2' not in fname.name:
            continue
        try:
            with h5py.File(fname, 'r') as f:
                rows.append(dict(name=fname.name, acq=str(f.attrs['acquisition']),
                                 shape=list(f['kspace'].shape)))
        except Exception as e:  # corrupt files are skipped, as in the repo
            print('skip', fname, e, file=sys.stderr)
    res[split] = rows
    print(split, len(rows), flush=True)
OUT.write_text(json.dumps(res, indent=1))
