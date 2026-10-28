"""CPU checks for the seeds/ES follow-up (run from the worktree root, PYTHONPATH = worktree):
1. "es" mode = fair_energy_score / sqrt(C H W); "rcgan" mode adds l1_std_p and updates beta.
2. Gradient scale of the ES/sqrt(D) regulariser vs the F regulariser (alpha_fair, alpha = 1) and MS, on real training
   truths with a Gaussian "posterior" (gens = x + s * noise), so ES is not negligible next to the critic term.
3. seed_everything(seed, workers=True) changes the generator init, and the worker-side numpy stream (observation noise).
"""
import json
import math
import pathlib
import types

import numpy as np
import torch
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.seed import pl_worker_init_function

from data.lightning.MassMappingDataModule import MMDataTransform
from models.lightning.mmGAN import mmGAN
from models.lightning.proper_scores import fair_energy_score, ms_fair_crps

cfg = json.loads(json.dumps(yaml.safe_load(open("configs/paper_mm.yml"))), object_hook=lambda d: types.SimpleNamespace(**d))

# 1. modes
m_es = mmGAN(cfg, "smoke", 1, crps_mode="es")
torch.manual_seed(0)
g, x = torch.randn(2, 2, 1, 300, 300), torch.randn(2, 1, 300, 300)
assert abs(m_es.crps_loss(g, x).item() - fair_energy_score(g, x).item() / 300.0) < 1e-6
m_rc = mmGAN(cfg, "smoke", 1, crps_mode="rcgan")
assert m_rc.std_mult == 1
print("modes ok: es", m_es.crps_loss(g, x).item(), "| rcgan l1_std_p", m_rc.l1_std_p(g.mean(1), g, x).item())

# 2. gradient scale on real truths (normalised as in training)
files = sorted((pathlib.Path(cfg.data_path) / "kappa_train").iterdir())[:4]
xt = torch.stack([torch.from_numpy((np.load(f, allow_pickle=True).astype(np.float64).real - cfg.kappa_mean) / cfg.kappa_std)
                  for f in files]).float().reshape(4, 1, 300, 300)
m_f = mmGAN(cfg, "smoke", 1, crps_mode="alpha_fair", crps_alpha=1.0)
m_ms = mmGAN(cfg, "smoke", 1, crps_mode="ms")
for s in (0.3, 0.6, 1.0):
    torch.manual_seed(1)
    out = {}
    for name, m in [("F", m_f), ("MS", m_ms), ("ES/sqrtD", m_es)]:
        gg = (xt.unsqueeze(1) + s * torch.randn(4, 2, 1, 300, 300)).requires_grad_(True)
        loss = m.crps_loss(gg, xt)
        loss.backward()
        out[name] = (loss.item(), gg.grad.norm().item())
        torch.manual_seed(1)
    print(f"s={s}: " + "  ".join(f"{k} loss {v[0]:.4f} |grad| {v[1]:.2e}" for k, v in out.items()))

# 3. seeds
def gen_params(seed):
    seed_everything(seed, workers=True)
    m = mmGAN(cfg, "smoke", 1, crps_mode="alpha_fair", crps_alpha=1.0)
    return torch.cat([p.detach().flatten() for p in m.generator.parameters()])


p0, p0b, p1 = gen_params(0), gen_params(0), gen_params(1)
print("init: seed0 == seed0", torch.equal(p0, p0b), "| seed0 == seed1", torch.equal(p0, p1),
      "| max |diff| 0 vs 1", (p0 - p1).abs().max().item())
assert torch.equal(p0, p0b) and not torch.equal(p0, p1)


class Noise(torch.utils.data.Dataset):
    """Draws the training observation noise exactly as MMDataTransform does (numpy global RNG, inside the worker)."""
    def __init__(self):
        self.tr = MMDataTransform(cfg, test=False)
        self.x = np.load(files[0], allow_pickle=True).astype(np.float64)

    def __len__(self):
        return 4

    def __getitem__(self, i):
        y = self.tr(self.x)[0]
        y = y.flatten()
        return y[y != 0][:5]  # skip masked (zeroed) pixels


def worker_noise(seed):
    seed_everything(seed, workers=True)
    dl = torch.utils.data.DataLoader(Noise(), batch_size=2, num_workers=2, worker_init_fn=pl_worker_init_function)
    return torch.cat([b for b in dl])


n0, n0b, n1 = worker_noise(0), worker_noise(0), worker_noise(1)
print("noise: seed0 == seed0", torch.allclose(n0, n0b), "| seed0 == seed1", torch.allclose(n0, n1))
assert torch.allclose(n0, n0b) and not torch.allclose(n0, n1)
print("ok")
