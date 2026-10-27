"""Check that cached items equal the repo's DataTransform output for a few val slices."""
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.lightning.MRIDataModule import DataTransform  # noqa: E402
from mri_crps.data import CachedSliceData  # noqa: E402

ds = CachedSliceData('val', with_maps=False)
tf = DataTransform(types.SimpleNamespace(im_size=384), test=True)
for i in [3, 50]:
    name, s = ds.examples[i]
    with h5py.File('/share/gpu0/tl3/fastMRI_dataset/multicoil_val/' + name, 'r') as f:
        k = f['kspace'][s]
    ref = tf(k, None, None, name, s, 1j * np.ones((8, 384, 384)))
    for a, b, n in zip(ref[:5], ds[i][:5], ['y', 'x', 'mask', 'mean', 'std']):
        a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
        print(name, s, n, tuple(a.shape), 'max abs diff', float((a - b).abs().max()), 'max abs', float(a.abs().max()))
