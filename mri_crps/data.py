"""Cached fastMRI dataset and data module for the CRPS paper experiments.

Items are identical to those of data/lightning/MRIDataModule.MRIDataModule:
(y, x, mask, mean, std, maps, fname, slice), with y the normalised zero-filled 16-channel
(8 coils x real/imag) input, x the normalised fully sampled 8-coil image, and maps the ESPIRiT
sensitivity maps (computed from the undersampled k-space, as in scripts/mri/estimate_maps.py).
The only difference is that the coil-compressed images are read from the cache written by
preprocess.py instead of being recomputed from the raw h5 k-space every time.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from utils.mri import transforms
from utils.mri.fftc import fft2c_new, ifft2c_new
from utils.mri.get_mask import get_mask

OUT = Path('/share/gpu0/jjwhit/crps_review/paper/mri')


def make_item(im, im_size=384):
    """im: complex (8, H, W) numpy array -> (y, x, mask, mean, std), as in DataTransform.__call__."""
    mask = get_mask(im_size)
    im_tensor = transforms.to_tensor(np.ascontiguousarray(im.transpose(1, 2, 0)).astype(np.complex128)).permute(2, 0, 1, 3)
    kspace = fft2c_new(im_tensor)
    masked_kspace = kspace * mask
    input_tensor = ifft2c_new(masked_kspace)

    normalized_input, mean, std = transforms.normalize_instance(input_tensor)
    normalized_gt = transforms.normalize(im_tensor, mean, std)

    final_input = torch.zeros(16, im_size, im_size)
    final_input[0:8] = normalized_input[..., 0]
    final_input[8:16] = normalized_input[..., 1]
    final_gt = torch.zeros(16, im_size, im_size)
    final_gt[0:8] = normalized_gt[..., 0]
    final_gt[8:16] = normalized_gt[..., 1]
    return final_input.float(), final_gt.float(), mask, mean, std


class CachedSliceData(Dataset):
    def __init__(self, split, n_slices=None, with_maps=True, max_volumes=None):
        splits = json.loads((OUT / 'manifest' / 'splits.json').read_text())
        entries = splits[split]
        if max_volumes is not None:
            entries = entries[:max_volumes]
        self.split = split
        self.with_maps = with_maps
        self.examples = []
        for e in entries:
            ns = e['n_slices'] if n_slices is None else min(n_slices, e['n_slices'])
            self.examples += [(e['name'], s) for s in range(ns)]
        self._cache = {}

    def __len__(self):
        return len(self.examples)

    def _load(self, kind, name):
        key = (kind, name)
        if key not in self._cache:
            self._cache[key] = np.load(OUT / kind / self.split / (name + '.npy'), mmap_mode='r')
        return self._cache[key]

    def __getitem__(self, i):
        name, s = self.examples[i]
        im = np.asarray(self._load('cache', name)[s])
        y, x, mask, mean, std = make_item(im)
        if self.with_maps and not (os.environ.get('MRI_ALLOW_NO_MAPS') and
                                   not (OUT / 'maps' / self.split / (name + '.npy')).exists()):
            maps = transforms.to_tensor(np.asarray(self._load('maps', name)[s]))
        else:
            # same fallback as the repo's training set (maps are not used for training)
            maps = transforms.to_tensor(1j * np.ones((8, 384, 384), dtype=np.complex64))
        return y, x, mask, mean, std, maps, name, s


class CachedMRIDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.prepare_data_per_node = True
        self.args = args

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train = CachedSliceData('train', with_maps=False, max_volumes=self.args.n_train_volumes)
        self.validate = CachedSliceData('val', with_maps=True)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers,
                          drop_last=True, pin_memory=False, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                          drop_last=True, pin_memory=False, persistent_workers=True)
