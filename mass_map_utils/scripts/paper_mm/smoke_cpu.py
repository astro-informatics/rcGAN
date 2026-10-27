"""CPU smoke test of mm_run.py: module provenance, checkpoint loading (old and new), one forward pass."""
import sys

import torch

import common
import models.lightning.mmGAN as mm

print("mmGAN from", mm.__file__)
print("data from", sys.modules["data.lightning.MassMappingDataModule"].__file__)
ts = common.TestSet(common.load_cfg())
y, x = ts.get(0)
for key, ck in [("F100", "checkpoint_best.ckpt"), ("MS", sys.argv[1])]:
    m = common.load_model(common.MODELS[key], ckpt=ck, device="cpu")
    torch.manual_seed(0)
    with torch.no_grad():
        s = m(y[None].expand(2, -1, -1, -1).contiguous())
    print(key, m.crps_mode, m.crps_alpha, getattr(m, "adv_scale", None), tuple(s.shape), float(s.std()))
print("ok")
