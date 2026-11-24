"""Per-pixel posterior shape and width vs noise for the MRI models (R=8), plus tail-rank diagnostics.

For every 4th test slice (54 of 216), N = 128 samples per model (same latent seeds as the main evaluation), fg mask:
- per-pixel sample skewness, Pearson kurtosis (Gaussian 3, uniform 1.8), Sarle bimodality coefficient (> 0.555 suggests
  bimodality/flat top), fraction of pixels whose sample range touches 0 (bounded magnitude);
- posterior sd / noise sd, where the noise sd per complex component is estimated from the GT magnitude in the
  background (support minus a dilated fg, Rayleigh: sigma = sqrt(mean(m^2) / 2));
- truth rank: fraction of pixels where the truth is outside the sample range (expected 2/(N+1)), the extreme 1% rank
  density, and central-50% coverage (flat-top signature: over-coverage at 50%, too many extreme ranks).
Results -> /share/gpu0/jjwhit/crps_review/candidates/mri_shapes.json.

Usage: python candidates/mri_shapes.py [--names R F MSsig] [--every 4] [--n 128]
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
res = {}
for name in args.names:
    model = rcGAN.load_from_checkpoint(str(OUT / 'ckpt' / 'final' / f'{name}.ckpt')).cuda().eval()
    acc = {k: [] for k in ['skew', 'kurt', 'bc', 'sd_over_noise', 'touch0']}
    rank_out, rank_lo1, rank_hi1, cov50, npx, noise = 0, 0, 0, 0, 0, []
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
            sm = F.conv2d(gt[None, None], box, padding=2)[0, 0]
            fg = sup & (sm > FG_THR)
            dil = F.max_pool2d(fg[None, None].float(), 21, stride=1, padding=10)[0, 0] > 0
            bg = sup & ~dil
            sig = float(torch.sqrt((gt[bg] ** 2).mean() / 2)) if bg.sum() > 500 else float('nan')
            noise.append(sig)
            torch.manual_seed(SEED0 + i)
            s = []
            for c in range(0, args.n, CHUNK):
                b = min(CHUNK, args.n - c)
                g = model(y.expand(b, -1, -1, -1), mask.expand(b, -1, -1, -1, -1))
                s.append(coil_combine(g, std, mean, maps) / scale)
            s = torch.cat(s)[:, fg].double()  # (N, P)
            t = gt[fg].double()
            m = s.mean(0)
            d = s - m
            v = (d ** 2).mean(0)
            sk = (d ** 3).mean(0) / v ** 1.5
            ku = (d ** 4).mean(0) / v ** 2
            n = s.shape[0]
            bc = (sk ** 2 + 1) / (ku - 3 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
            acc['skew'].append(sk.cpu().numpy())
            acc['kurt'].append(ku.cpu().numpy())
            acc['bc'].append(bc.cpu().numpy())
            acc['sd_over_noise'].append((s.std(0) / sig).cpu().numpy())
            acc['touch0'].append((s.min(0).values <= 1e-3).float().cpu().numpy())
            below = (s < t).sum(0)
            rank_out += int(((below == 0) | (below == n)).sum())
            r = (below.double() + torch.rand_like(t)) / (n + 1)
            rank_lo1 += int((r < 0.01).sum())
            rank_hi1 += int((r > 0.99).sum())
            cov50 += int(((r - 0.5).abs() <= 0.25).sum())
            npx += int(fg.sum())
    q = lambda a: [float(np.quantile(a, v)) for v in (0.1, 0.25, 0.5, 0.75, 0.9)]  # noqa: E731
    cat = {k: np.concatenate(v) for k, v in acc.items()}
    res[name] = dict(
        n_slices=len(items), n_pixels=npx, N=args.n, quantiles='10/25/50/75/90%',
        skew=q(cat['skew']), kurt_pearson=q(cat['kurt']), bimodality_coef=q(cat['bc']),
        frac_bc_gt_0555=float((cat['bc'] > 0.555).mean()), frac_kurt_lt_2p5=float((cat['kurt'] < 2.5).mean()),
        frac_kurt_gt_4=float((cat['kurt'] > 4).mean()), frac_abs_skew_gt_1=float((np.abs(cat['skew']) > 1).mean()),
        sd_over_noise=q(cat['sd_over_noise']), frac_touch0=float(cat['touch0'].mean()),
        noise_sigma_median=float(np.nanmedian(noise)),
        truth_outside_range=rank_out / npx, truth_outside_range_expected=2 / (args.n + 1),
        rank_density_lowest1=rank_lo1 / npx / 0.01, rank_density_highest1=rank_hi1 / npx / 0.01,
        cov50_minus_nominal=cov50 / npx - 0.5)
    print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(DST / 'mri_shapes.json', 'w'), indent=1)
