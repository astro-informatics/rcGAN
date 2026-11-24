"""Conditional (local) calibration of the MRI models: is the calibration right everywhere, or only on average?

rcGAN tunes one global beta, which fixes the *average* predictive spread; a proper per-pixel score is minimised by the
true conditional law at every pixel. So stratify the truth's randomised PIT by features that are known without the truth:
- the model's own predictive sd (deciles over all fg pixels of all slices),
- posterior-mean intensity (deciles),
- local edge strength |grad posterior mean| (deciles),
- the slice (per-slice spread-skill, 54 slices).
Per stratum: 50% / 90% central coverage minus nominal and spread-skill v(1+1/N)/e.
Summary "conditional calibration error" = RMS over the 10 deciles of (cov90 - 0.9), and of log(spread-skill).
Same 54 slices / seeds as mri_shapes.py. -> /share/gpu0/jjwhit/crps_review/candidates/mri_condcov.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.lightning.rcGAN import rcGAN  # noqa: E402
from mri_crps.data import OUT, CachedSliceData  # noqa: E402
from mri_crps.sample import CHUNK, SEED0, coil_combine  # noqa: E402

DST = Path('/share/gpu0/jjwhit/crps_review/candidates')
FG_THR = 0.05

p = argparse.ArgumentParser()
p.add_argument('--names', nargs='+', default=['R', 'F', 'MSsig'])
p.add_argument('--every', type=int, default=4)
p.add_argument('--n', type=int, default=128)
args = p.parse_args()
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

ds = CachedSliceData('test', with_maps=True)
items = list(range(0, len(ds), args.every))
box = torch.ones(1, 1, 5, 5, device='cuda') / 25
gen = torch.Generator(device='cuda').manual_seed(0)


def strat(feat, pit, err2, var, N, nb=10):
    edges = np.quantile(feat, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(edges, feat, side='right') - 1, 0, nb - 1)
    out = dict(cov50=[], cov90=[], spread_skill=[])
    for k in range(nb):
        s = b == k
        out['cov50'].append(float((np.abs(pit[s] - 0.5) <= 0.25).mean() - 0.5))
        out['cov90'].append(float((np.abs(pit[s] - 0.5) <= 0.45).mean() - 0.9))
        out['spread_skill'].append(float(var[s].mean() * (1 + 1 / N) / err2[s].mean()))
    out['cond_ce_cov90'] = float(np.sqrt(np.mean(np.square(out['cov90']))))
    out['cond_ce_logss'] = float(np.sqrt(np.mean(np.square(np.log(out['spread_skill'])))))
    return out


res = {}
for name in args.names:
    model = rcGAN.load_from_checkpoint(str(OUT / 'ckpt' / 'final' / f'{name}.ckpt')).cuda().eval()
    cols = {k: [] for k in ['sd', 'mean', 'edge', 'pit', 'err2', 'var']}
    per_slice = []
    with torch.no_grad():
        for i in items:
            y, x, mask, mean, std, maps, _, _ = ds[i]
            y, x, mask = y.cuda()[None], x.cuda()[None], mask.cuda()[None]
            mean, std = float(mean), float(std)
            maps = torch.complex(maps[..., 0], maps[..., 1]).cuda()
            gt = coil_combine(x, std, mean, maps)[0]
            scale = float(gt.max())
            gt = gt / scale
            sup = (maps.abs() ** 2).sum(0) > 0
            fg = sup & (F.conv2d(gt[None, None], box, padding=2)[0, 0] > FG_THR)
            torch.manual_seed(SEED0 + i)
            s = []
            for c in range(0, args.n, CHUNK):
                b = min(CHUNK, args.n - c)
                g = model(y.expand(b, -1, -1, -1), mask.expand(b, -1, -1, -1, -1))
                s.append(coil_combine(g, std, mean, maps) / scale)
            s = torch.cat(s).double()
            m = s.mean(0)
            gy, gx = torch.gradient(m)
            edge = torch.sqrt(gx ** 2 + gy ** 2)
            v = s.var(0)
            below = (s < gt.double()).sum(0).double()
            pit = (below + torch.rand(below.shape, device='cuda', generator=gen, dtype=torch.float64)) / (args.n + 1)
            e2 = (m - gt) ** 2
            for k, t in zip(['sd', 'mean', 'edge', 'pit', 'err2', 'var'], [v.sqrt(), m, edge, pit, e2, v]):
                cols[k].append(t[fg].cpu().numpy())
            per_slice.append(float(v[fg].mean() * (1 + 1 / args.n) / e2[fg].mean()))
    c = {k: np.concatenate(v) for k, v in cols.items()}
    N = args.n
    res[name] = dict(
        n_slices=len(items), n_pixels=int(len(c['pit'])),
        overall=dict(cov50=float((np.abs(c['pit'] - 0.5) <= 0.25).mean() - 0.5),
                     cov90=float((np.abs(c['pit'] - 0.5) <= 0.45).mean() - 0.9),
                     spread_skill=float(c['var'].mean() * (1 + 1 / N) / c['err2'].mean())),
        by_sd=strat(c['sd'], c['pit'], c['err2'], c['var'], N),
        by_intensity=strat(c['mean'], c['pit'], c['err2'], c['var'], N),
        by_edge=strat(c['edge'], c['pit'], c['err2'], c['var'], N),
        per_slice_spread_skill=dict(values=per_slice, rms_log=float(np.sqrt(np.mean(np.log(per_slice) ** 2))),
                                    q10_50_90=[float(np.quantile(per_slice, v)) for v in (0.1, 0.5, 0.9)]))
    print(name, json.dumps({k: res[name][k] for k in ['overall']}),
          {k: (round(res[name][k]['cond_ce_cov90'], 3), round(res[name][k]['cond_ce_logss'], 3)) for k in ['by_sd', 'by_intensity', 'by_edge']},
          'per-slice rms log ss', round(res[name]['per_slice_spread_skill']['rms_log'], 3), flush=True)
    json.dump(res, open(DST / 'mri_condcov.json', 'w'), indent=1)
