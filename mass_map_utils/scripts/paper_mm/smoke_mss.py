"""CPU check of the ms_sigma CRPS mode: loss = ms_fair_crps with the saved 1/sigma_s weights; weights land in hparams."""
import json, types, yaml, torch
from models.lightning.mmGAN import mmGAN
from models.lightning.proper_scores import ms_fair_crps, fair_crps
cfg = json.loads(json.dumps(yaml.safe_load(open("configs/paper_mm.yml"))), object_hook=lambda d: types.SimpleNamespace(**d))
w = json.load(open("/share/gpu0/jjwhit/crps_review/paper/mm/ms_sigma_levels.json"))["weights"]
m = mmGAN(cfg, "smoke", 1, crps_mode="ms_sigma", ms_level_weights=w)
torch.manual_seed(0)
g, x = torch.randn(2, 2, 1, 300, 300), torch.randn(2, 1, 300, 300)
a = m.crps_loss(g, x).item()
b = ms_fair_crps(g, x, 7, w).item()
print("ms_sigma", a, b, "unweighted", ms_fair_crps(g, x).item(), "pixel", fair_crps(g, x).item())
assert abs(a - b) < 1e-9 and m.hparams.ms_level_weights == w
print("ok")
