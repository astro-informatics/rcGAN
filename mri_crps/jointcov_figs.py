"""Figures from jointcov.json (shared jointcov.py schema) -> <out>/figures/jointcov/*.{png,pdf}, plus a markdown table.

(i) marginal (pixel) coverage - nominal vs nominal; (ii) full-image coverage curves for maxz / depth / avgrank (and mean);
(iii) 90% coverage error vs region size for each statistic. Grey: range of the null-check curves over methods
(truth replaced by a sample; exchangeable by construction), i.e. the Monte Carlo band including spatial correlation.

Usage: python mri_crps/jointcov_figs.py [--json .../jointcov.json] [--tag ''] > table.md
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path('/share/gpu0/jjwhit/crps_review/paper/mri')
COL = {'R': 'tab:red', 'F': 'tab:blue', 'MS': 'tab:green', 'MSsig': 'tab:purple', 'EXACT': 'k'}
STATS = ['pixel', 'mean', 'maxz', 'depth', 'avgrank']
TITLE = {'pixel': 'pixel (marginal)', 'mean': 'region mean', 'maxz': 'max |z| (simultaneous band)',
         'depth': 'depth (energy-score geometry)', 'avgrank': 'average rank (dependence)'}

ap = argparse.ArgumentParser()
ap.add_argument('--json', default=str(OUT / 'jointcov.json'))
ap.add_argument('--tag', default='')
args = ap.parse_args()
D = json.load(open(args.json))
M = D['methods']
names = [k for k in ['R', 'F', 'MS', 'MSsig', 'EXACT'] if k in M] + [k for k in M if k not in COL]
lv = np.array(next(iter(M.values()))['model']['levels'])
i90 = int(np.argmin(np.abs(lv - 0.9)))
FIG = OUT / 'figures' / 'jointcov'
FIG.mkdir(parents=True, exist_ok=True)


def cov(k, part, st, key):
    e = M[k][part]['stats'][st].get(key)
    return None if e is None else np.array(e['cov'])


def null_band(st, key):
    c = [cov(k, 'null', st, key) for k in names if M[k].get('null')]
    c = [x for x in c if x is not None]
    return (np.min(c, 0) - lv, np.max(c, 0) - lv) if c else None


def save(fig, name):
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG / f'{name}{args.tag}.{ext}', dpi=150)
    plt.close(fig)


def curve_panel(ax, st, key, title):
    b = null_band(st, key)
    if b is not None:
        ax.fill_between(lv, b[0], b[1], color='0.75', lw=0, label='null check (range)')
    for k in names:
        c = cov(k, 'model', st, key)
        if c is not None:
            ax.plot(lv, c - lv, color=COL.get(k), marker='o', ms=2.5, label=D['labels'].get(k, k) + f' ({k})')
    ax.axhline(0, color='k', lw=0.8)
    ax.set(xlabel='nominal level', ylabel='empirical - nominal', title=title)


# (i) marginal coverage
fig, ax = plt.subplots(figsize=(5.2, 3.8))
curve_panel(ax, 'pixel', '1', f"per-pixel coverage ({D['n_cases']} slices, N={D['N']})")
ax.legend(fontsize=7)
save(fig, 'marginal')

# (ii) full-image curves
fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
for ax, st in zip(axs, ['maxz', 'depth', 'avgrank', 'mean']):
    curve_panel(ax, st, 'full', f'full image: {TITLE[st]}')
axs[0].legend(fontsize=7)
save(fig, 'full_image')

# (iii) 90% error vs region size
keys = [k for k in next(iter(M.values()))['model']['stats']['mean'] if k != 'full']
sizes = sorted(int(k) for k in keys)
xfull = sizes[-1] * 2.5
fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
for ax, st in zip(axs, ['mean', 'maxz', 'depth', 'avgrank']):
    ks = [str(s) for s in sizes if str(s) in M[names[0]]['model']['stats'][st]] + ['full']
    xs = [int(k) if k != 'full' else xfull for k in ks]
    lo = [null_band(st, k)[0][i90] for k in ks]
    hi = [null_band(st, k)[1][i90] for k in ks]
    ax.fill_between(xs, lo, hi, color='0.75', lw=0)
    for k in names:
        ax.plot(xs, [cov(k, 'model', st, kk)[i90] - lv[i90] for kk in ks], color=COL.get(k), marker='o', ms=3, label=k)
    ax.axhline(0, color='k', lw=0.8)
    ax.set(xscale='log', xlabel='region side (px); rightmost = full image', title=TITLE[st])
    ax.set_xticks(xs)
    ax.set_xticklabels([k for k in ks], fontsize=7)
    ax.minorticks_off()
axs[0].set_ylabel(f'coverage - nominal at {lv[i90]:.0%}')
axs[0].legend(fontsize=7)
save(fig, 'error_vs_size')

# markdown table: 90% coverage error (model; null in brackets is the largest |null error| over methods)
allkeys = ['1'] + [str(s) for s in sizes if s > 1] + ['full']
print(f"90% coverage - 0.90 ({D['n_cases']} slices, N = {D['N']}); last column per statistic block = max |null error| over methods\n")
print('| statistic | region | ' + ' | '.join(names) + ' | null max abs |')
print('|---|---|' + '---|' * (len(names) + 1))
for st in STATS:
    for kk in allkeys:
        if cov(names[0], 'model', st, kk) is None:
            continue
        nb = null_band(st, kk)
        nmax = max(abs(nb[0][i90]), abs(nb[1][i90])) if nb else float('nan')
        print(f'| {st} | {kk} | ' + ' | '.join(f'{cov(k, "model", st, kk)[i90] - lv[i90]:+.3f}' for k in names) + f' | {nmax:.3f} |')
