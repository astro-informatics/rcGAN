"""Aggregate per-slice statistics (stats/<name>.npz) into results.json + figures, with paired bootstrap CIs.

Bootstrap: B resamples of test slices (seed 0), the same resampled indices for every method, so differences
vs R get CIs. Ratio statistics are ratios of sums over the resampled slices.

Usage: python mri_crps/aggregate.py R F MS [--ref R] [--tag main]
"""
import argparse
import glob
import json
import os
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mri_crps.data import OUT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('names', nargs='+')
ap.add_argument('--ref', default='R')
ap.add_argument('--tag', default='main')
ap.add_argument('--B', type=int, default=2000)
ap.add_argument('--B-cfid', type=int, default=200)
args = ap.parse_args()

Z = {k: dict(np.load(OUT / 'stats' / f'{k}.npz')) for k in args.names}
z0 = Z[args.names[0]]
n = len(z0['psnr_P1'])
N = int(z0['N'])
LEVELS = z0['levels']
EXACT = z0['exact']
rng = np.random.default_rng(0)
C = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(args.B)]).astype(float)  # (B, n) counts
COL = {'R': 'tab:red', 'F': 'tab:blue', 'MS': 'tab:green', 'MSsig': 'tab:purple', 'S': 'tab:orange'}
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)


def ci(d):
    return np.nanpercentile(d, 2.5, axis=0), np.nanpercentile(d, 97.5, axis=0)


def boot_mean(a):
    """a (n, ...) per-slice values -> point, draws (B, ...) of the mean over slices (nan-aware)."""
    a = np.asarray(a, float)
    ok = ~np.isnan(a)
    a0 = np.where(ok, a, 0)
    w = ok.astype(float)
    pt = a0.sum(0) / w.sum(0)
    flat0, flatw = a0.reshape(n, -1), w.reshape(n, -1)
    d = (C @ flat0) / (C @ flatw)
    return pt, d.reshape((args.B,) + a.shape[1:])


def boot_ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    pt = num.sum(0) / den.sum(0)
    d = (C @ num.reshape(n, -1)) / (C @ den.reshape(n, -1))
    return pt, d.reshape((args.B,) + num.shape[1:])


def summ(fn, nd=4):
    """fn(z) -> (point, draws). Returns {name: {val, lo, hi, diff, diff_lo, diff_hi}}."""
    out, draws = {}, {}
    for k in args.names:
        pt, d = fn(Z[k])
        draws[k] = (pt, d)
    for k in args.names:
        pt, d = draws[k]
        lo, hi = ci(d)
        e = dict(val=np.round(pt, nd).tolist(), lo=np.round(lo, nd).tolist(), hi=np.round(hi, nd).tolist())
        if k != args.ref and args.ref in draws:
            pr, dr = draws[args.ref]
            dlo, dhi = ci(d - dr)
            e.update(diff=np.round(pt - pr, nd).tolist(), diff_lo=np.round(dlo, nd).tolist(), diff_hi=np.round(dhi, nd).tolist())
        out[k] = e
    return out, draws


def rank_ce(h):
    """h (..., nb) counts -> RMS deviation of the normalised histogram density from 1."""
    f = h / h.sum(-1, keepdims=True)
    nb = h.shape[-1]
    return np.sqrt(((nb * f - 1) ** 2).mean(-1))


def boot_rank_ce(h):
    h = np.asarray(h, float)
    pt = rank_ce(h.sum(0))
    d = rank_ce(np.einsum('bn,n...->b...', C, h))
    return pt, d


RES = dict(n_slices=n, N=N, levels=LEVELS.tolist(), exact_levels=EXACT.tolist(), methods=args.names, ref=args.ref,
           bootstrap=dict(B=args.B, seed=0, unit='test slice'))

# ---------------------------------------------------------------- 1. reconstruction
rec = {}
for P in [1, 2, 4, 8, 32]:
    for met in ['psnr', 'ssim', 'lpips']:
        rec[f'{met}_P{P}'] = summ(lambda z: boot_mean(z[f'{met}_P{P}']))[0]
rec['apsd32'] = summ(lambda z: boot_mean(z['apsd32']), 6)[0]
rec['psnr_zero_filled'] = summ(lambda z: boot_mean(z['psnr_zf']))[0]
# the 72 slices of the rcGAN paper's test set (first 12 volumes x 6 slices)
n72 = min(72, n)
rec['rcgan_testset_72'] = {k: dict(psnr_P1=float(np.mean(Z[k]['psnr_P1'][:n72])), psnr_P8=float(np.mean(Z[k]['psnr_P8'][:n72])),
                                   psnr_P32=float(np.mean(Z[k]['psnr_P32'][:n72])), ssim_P8=float(np.mean(Z[k]['ssim_P8'][:n72])),
                                   lpips_P8=float(np.mean(Z[k]['lpips_P8'][:n72]))) for k in args.names}
RES['reconstruction'] = rec

# CFID (repo implementation, evaluation_scripts/cfid/cfid_metric.py), 32 samples per test slice
from evaluation_scripts.cfid.cfid_metric import CFIDMetric  # noqa: E402

cfid_obj = CFIDMetric(gan=None, loader=None, image_embedding=None, condition_embedding=None, args=types.SimpleNamespace())


def cfid(z, idx):
    g = torch.from_numpy(z['emb_gen'][idx]).double()  # (k, 32, D)
    P = g.shape[1]
    t = torch.from_numpy(z['emb_true'][idx]).double()[:, None].expand(-1, P, -1)
    c = torch.from_numpy(z['emb_cond'][idx]).double()[:, None].expand(-1, P, -1)
    D = g.shape[-1]
    v, m, cc = cfid_obj.get_cfid_torch_pinv(y_predict=g.reshape(-1, D), x_true=c.reshape(-1, D), y_true=t.reshape(-1, D))
    return float(v), float(m), float(cc)


cf = {}
rng_c = np.random.default_rng(1)
cidx = [rng_c.integers(0, n, n) for _ in range(args.B_cfid)]
cf_draws = {}
for k in args.names:
    v, m, cc = cfid(Z[k], np.arange(n))
    d = np.array([cfid(Z[k], ix)[0] for ix in cidx])
    cf_draws[k] = d
    cf[k] = dict(cfid=v, m_comp=m, c_comp=cc, lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                 cfid_rcgan72=cfid(Z[k], np.arange(n72))[0])
for k in args.names:
    if k != args.ref:
        dd = cf_draws[k] - cf_draws[args.ref]
        cf[k].update(diff=cf[k]['cfid'] - cf[args.ref]['cfid'], diff_lo=float(np.percentile(dd, 2.5)), diff_hi=float(np.percentile(dd, 97.5)))
RES['cfid'] = dict(note='VGG16 avg-pool embeddings of min-max-normalised magnitude images, 32 samples per slice, '
                        'condition = zero-filled magnitude; bootstrap over slices (B=%d)' % args.B_cfid, **cf)

# ---------------------------------------------------------------- 2. marginal calibration
cal = {}
cal['pix_cov_minus_exact'] = summ(lambda z: (lambda p, d: (p - EXACT, d - EXACT))(*boot_mean(z['pix_cov'])))[0]
cal['pix_rank_ce'] = summ(lambda z: boot_rank_ce(z['pix_rankhist']))[0]
cal['pix_rankhist_density'] = {k: (10 * Z[k]['pix_rankhist'].sum(0) / Z[k]['pix_rankhist'].sum()).round(4).tolist() for k in args.names}
cal['pix_cov_abs_err_mean'] = summ(lambda z: (lambda p, d: (np.abs(p - EXACT).mean(), np.abs(d - EXACT).mean(1)))(*boot_mean(z['pix_cov'])))[0]
cal['bg_cov_minus_exact'] = summ(lambda z: (lambda p, d: (p - EXACT, d - EXACT))(*boot_mean(z['bg_cov'])))[0]
RES['marginal_calibration'] = cal

# ---------------------------------------------------------------- 3. joint calibration
jc = {}
for s in [4, 8, 16, 32, 64]:
    jc[f'patch{s}_cov_minus_exact'] = summ(lambda z: (lambda p, d: (p - EXACT, d - EXACT))(*boot_mean(z[f'patch{s}_cov'])))[0]
    jc[f'patch{s}_rank_ce'] = summ(lambda z: boot_rank_ce(z[f'patch{s}_rankhist']))[0]
    jc[f'patch{s}_var_corr_over_err2'] = summ(lambda z: boot_ratio(z[f'patch{s}_var'] * (1 + 1 / N), z[f'patch{s}_err2']))[0]
    jc[f'patch{s}_tiles_per_slice'] = {k: float(np.mean(Z[k][f'patch{s}_n'])) for k in args.names}
jc['global_cov_minus_exact'] = summ(lambda z: (lambda p, d: (p - EXACT, d - EXACT))(*boot_mean(z['global_cov'])))[0]


def global_rank_ce(z):
    r = z['global_rank']
    h = np.zeros((n, 10))
    h[np.arange(n), (r * 10) // (N + 1)] = 1
    return boot_rank_ce(h)


jc['global_rank_ce'] = summ(global_rank_ce)[0]
ke = z0['kedges']
jc['k_bins_edges_cycles_per_fov'] = ke.tolist()
jc['spec_sample_over_truth'] = summ(lambda z: boot_ratio(z['bp_s_mean'], z['bp_x']))[0]
jc['spec_postmean_over_truth'] = summ(lambda z: boot_ratio(z['bp_m'], z['bp_x']))[0]
jc['spec_err_over_spread'] = summ(lambda z: boot_ratio(z['bp_res_x'] / (1 + 1 / N), z['bp_res_s'] * N / (N - 1)))[0]


def bp_rank_ce(z):
    r = z['bp_rank']  # (n, K)
    K = r.shape[1]
    h = np.zeros((n, K, 10))
    for b in range(K):
        h[np.arange(n), b, (r[:, b] * 10) // (N + 1)] = 1
    return boot_rank_ce(h)


jc['bandpower_rank_ce'] = summ(bp_rank_ce)[0]
HI = ke[:-1] >= 32
jc['highk_def'] = 'radial bins with lower edge >= 32 cycles/FOV (of 192)'
jc['highk_sample_over_truth'] = summ(lambda z: boot_ratio(z['bp_s_mean'][:, HI].sum(1), z['bp_x'][:, HI].sum(1)))[0]
jc['highk_postmean_over_truth'] = summ(lambda z: boot_ratio(z['bp_m'][:, HI].sum(1), z['bp_x'][:, HI].sum(1)))[0]
jc['es_fair'] = summ(lambda z: boot_mean(z['es_fair']), 5)[0]
jc['vs_local_mean_over_lags'] = summ(lambda z: boot_mean(z['vs_local'].mean(1)), 7)[0]
RES['joint_calibration'] = jc

# ---------------------------------------------------------------- 4. diversity / collapse
dv = {}
dv['spread_skill_v_corr_over_e'] = summ(lambda z: boot_ratio(z['var_sum'] * (1 + 1 / N), z['err2_mean']))[0]
dv['spread_skill_def'] = 'v(1+1/N)/e, v = mean posterior variance, e = MSE of the N-sample mean, fg pixels; 1 = calibrated, <1 under-dispersed'
dv['pairdist_over_truthdist'] = summ(lambda z: boot_ratio(z['pair_dbar'], np.sqrt(2 / (1 + 1 / N)) * z['err_mean_norm']))[0]
dv['frac_pairs_closer_than_half_mean'] = summ(lambda z: boot_mean(z['frac_pairs_lt_half_mean']))[0]
dv['min_pair_over_mean_pair'] = summ(lambda z: boot_mean(z['min_pair_over_mean']))[0]
dv['highk_sample_over_truth'] = jc['highk_sample_over_truth']
RES['diversity'] = dv


# ---------------------------------------------------------------- training logs
def read_log(k):
    f = OUT / 'ckpt' / 'logs' / k / 'version_0' / 'metrics.csv'
    return pd.read_csv(f) if f.exists() else None


tr = {}
for k in args.names:
    df = read_log(k)
    if df is None:
        continue
    v = df.dropna(subset=['val_psnr_8']) if 'val_psnr_8' in df else None
    tr[k] = dict(final_beta=float(v['beta_std'].iloc[-1]) if v is not None and len(v) else None,
                 final_val_psnr_8=float(v['val_psnr_8'].iloc[-1]) if v is not None and len(v) else None,
                 final_val_psnr_1=float(v['val_psnr_1'].iloc[-1]) if v is not None and len(v) else None,
                 final_val_psnr_gain_minus_2p5=float(-v['val_psnr_diff'].iloc[-1]) if v is not None and len(v) else None,
                 epochs=int(df['epoch'].max()) + 1, steps=int(df['step'].max()),
                 nonfinite_g_loss=int((~np.isfinite(df['g_loss'].dropna())).sum()) if 'g_loss' in df else None)
RES['training'] = tr

json.dump(RES, open(OUT / f'results_{args.tag}.json', 'w'), indent=1)
print(json.dumps({k: RES[k] for k in ['cfid', 'training']}, indent=1))


# ---------------------------------------------------------------- figures
def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f'{name}_{args.tag}.png', dpi=150)
    fig.savefig(FIG / f'{name}_{args.tag}.pdf')
    plt.close(fig)


def errline(ax, x, e, k, **kw):
    v, lo, hi = np.array(e['val']), np.array(e['lo']), np.array(e['hi'])
    ax.plot(x, v, color=COL.get(k.split('_')[0]), marker='o', ms=3, label=k, **kw)
    ax.fill_between(x, lo, hi, color=COL.get(k.split('_')[0]), alpha=0.2, lw=0)


fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for k in args.names:
    errline(axs[0], EXACT, cal['pix_cov_minus_exact'][k], k)
    axs[1].bar(np.arange(10) + 0.25 * (args.names.index(k) - 1), cal['pix_rankhist_density'][k], width=0.25, color=COL.get(k.split('_')[0]), label=k)
    errline(axs[2], EXACT, jc['patch16_cov_minus_exact'][k], k)
axs[0].axhline(0, color='k', lw=0.8)
axs[0].set(xlabel='nominal level', ylabel='coverage - nominal', title='per-pixel coverage (fg mask)')
axs[1].axhline(1, color='k', lw=0.8)
axs[1].set(xlabel='rank decile of truth', ylabel='density', title='per-pixel rank histogram')
axs[2].axhline(0, color='k', lw=0.8)
axs[2].set(xlabel='nominal level', ylabel='coverage - nominal', title='16x16 patch-mean coverage')
axs[0].legend()
save(fig, 'calibration')

fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
sizes = [1, 4, 8, 16, 32, 64]
for k in args.names:
    for j, (lvl, ax) in enumerate(zip([2, 5], axs)):
        vals = [cal['pix_cov_minus_exact'][k]] + [jc[f'patch{s}_cov_minus_exact'][k] for s in sizes[1:]]
        e = dict(val=[v['val'][lvl] for v in vals], lo=[v['lo'][lvl] for v in vals], hi=[v['hi'][lvl] for v in vals])
        errline(ax, sizes, e, k)
        ax.set(xscale='log', xlabel='patch size (px)', ylabel='coverage - nominal', title=f'patch-mean coverage, {LEVELS[lvl]:.0%} intervals')
        ax.axhline(0, color='k', lw=0.8)
axs[0].legend()
save(fig, 'patch_coverage')

kc = np.sqrt(ke[:-1] * ke[1:])
kc[0] = 0.5
fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
for k in args.names:
    errline(axs[0], kc, jc['spec_sample_over_truth'][k], k)
    errline(axs[1], kc, jc['spec_postmean_over_truth'][k], k)
    errline(axs[2], kc, jc['spec_err_over_spread'][k], k)
for ax, t in zip(axs, ['per-sample / truth power', 'posterior-mean / truth power', 'error / spread power (1 = calibrated)']):
    ax.set(xscale='log', xlabel='k (cycles / FOV)', title=t)
    ax.axhline(1, color='k', lw=0.8)
axs[0].legend()
save(fig, 'spectra')

fig, ax = plt.subplots(figsize=(4.5, 3.4))
for k in args.names:
    errline(ax, [1, 2, 4, 8, 32], dict(val=[rec[f'psnr_P{P}'][k]['val'] for P in [1, 2, 4, 8, 32]],
                                       lo=[rec[f'psnr_P{P}'][k]['lo'] for P in [1, 2, 4, 8, 32]],
                                       hi=[rec[f'psnr_P{P}'][k]['hi'] for P in [1, 2, 4, 8, 32]]), k)
ax.set(xscale='log', xlabel='P (samples averaged)', ylabel='PSNR (dB)', xticks=[1, 2, 4, 8, 32], xticklabels=[1, 2, 4, 8, 32])
ax.legend()
save(fig, 'psnr_vs_P')

# training curves
fig, axs = plt.subplots(1, 4, figsize=(17, 3.4))
for k in args.names:
    df = read_log(k)
    if df is None:
        continue
    g = df.dropna(subset=['g_loss'])
    axs[0].plot(g['step'], g['g_reg'].rolling(50, min_periods=1).mean(), color=COL.get(k.split('_')[0]), label=k)
    d = df.dropna(subset=['d_loss'])
    axs[1].plot(d['step'], d['d_loss'].rolling(50, min_periods=1).mean(), color=COL.get(k.split('_')[0]), label=k)
    v = df.dropna(subset=['val_psnr_8'])
    axs[2].plot(v['epoch'], v['val_psnr_8'], color=COL.get(k.split('_')[0]), label=f'{k} P=8')
    axs[2].plot(v['epoch'], v['val_psnr_1'], color=COL.get(k.split('_')[0]), ls='--', label=f'{k} P=1')
    axs[3].plot(v['epoch'], v['beta_std'], color=COL.get(k.split('_')[0]), label=k)
axs[0].set(title='generator regulariser (rolling mean)', xlabel='step')
axs[1].set(title='critic loss (rolling mean)', xlabel='step')
axs[2].set(title='validation PSNR', xlabel='epoch')
axs[3].set(title='rcGAN beta (fixed for F, MS)', xlabel='epoch')
axs[2].legend(fontsize=7)
save(fig, 'training')
print('figures in', FIG)
