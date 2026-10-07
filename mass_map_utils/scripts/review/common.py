"""Shared helpers for the CRPS review (Stage A) scripts.

Everything here works in the generator's *native* (B, 1, H, W) orientation: no
`reformat` (which transposes H/W) and no 180-degree rotation, so the COSMOS mask
applies directly to samples and truth.

Observations are regenerated deterministically: map i of the sorted test set
always gets the same shear-noise realisation (seeded by i), so every model is
evaluated on identical inputs and per-map comparisons are paired.
"""
import json
import os
import pathlib
import sys
import types

import numpy as np
import torch
import yaml

REPO = str(pathlib.Path(__file__).resolve().parents[3])
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from data.lightning.MassMappingDataModule import MMDataTransform  # noqa: E402

CKPT_DIR = "/share/gpu0/jjwhit/mass_map/mm_models/"
TEST_DIR = "/share/gpu0/jjwhit/kappa_cosmos_simulations/cropped_dataset/kappa_test"
OUT_ROOT = "/share/gpu0/jjwhit/crps_review/"
CONFIG = os.path.join(REPO, "configs", "crps_eval", "mass_map_review.yml")

MODELS = {
    "R": "mmgan_training_real_output",
    "F3": "mmgan_crps_fair3",
    "F100": "mmgan_crps_alpha_100",
    "FB": "mmgan_crps_fair2",
    "S": "mmgan_crps_base",
    "NL": "mmgan_no_loss",
}
NOISE_SEED_OFFSET = 1_000_000  # observation noise seed = offset + map index


def load_cfg(path=CONFIG):
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return json.loads(json.dumps(cfg), object_hook=lambda d: types.SimpleNamespace(**d))


def load_model(exp, ckpt="checkpoint_best.ckpt", device="cuda"):
    from models.lightning.mmGAN import mmGAN

    model = mmGAN.load_from_checkpoint(checkpoint_path=os.path.join(CKPT_DIR, exp, ckpt), map_location="cpu")
    return model.to(device).eval()


class TestSet:
    """Sorted kappa_test files with fixed per-map observation noise."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.files = sorted(pathlib.Path(TEST_DIR).iterdir())
        self.transform = MMDataTransform(cfg, test=True)
        self.mask = self.transform.mask.astype(bool)  # native orientation, (H, W)

    def __len__(self):
        return len(self.files)

    def get(self, i):
        kappa = np.load(self.files[i], allow_pickle=True).astype(np.float64)
        state = np.random.get_state()
        np.random.seed(NOISE_SEED_OFFSET + i)
        y, x, _, _ = self.transform(kappa)
        np.random.set_state(state)
        return y, x  # y: (4, H, W) normalised obs; x: (1, H, W) normalised truth

    def batch(self, idx):
        ys, xs = zip(*(self.get(i) for i in idx))
        return torch.stack(ys), torch.stack(xs)


def to_kappa(t, cfg):
    return t * cfg.kappa_std + cfg.kappa_mean


@torch.no_grad()
def sample(model, y, n, chunk=64, seed=None):
    """n posterior samples for a single observation y (4, H, W) -> (n, H, W) in kappa units."""
    if seed is not None:
        torch.manual_seed(seed)
    chunk = min(chunk, _CHUNK.get("max", chunk))
    out, done = [], 0
    y = y.to(model.device)
    while done < n:
        k = min(chunk, n - done)
        try:
            out.append(model(y.unsqueeze(0).expand(k, -1, -1, -1).contiguous())[:, 0])
            done += k
        except torch.cuda.OutOfMemoryError:
            if chunk == 1:
                raise
            torch.cuda.empty_cache()
            chunk = max(1, chunk // 2)
            print(f"OOM: reducing sampling chunk to {chunk}", flush=True)
    _CHUNK["max"] = chunk
    return to_kappa(torch.cat(out), model.args)


_CHUNK = {}  # remembers reduced chunk sizes after OOM
