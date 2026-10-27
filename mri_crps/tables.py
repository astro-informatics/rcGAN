"""Markdown tables from results_<tag>.json (value [95% CI]; diff vs ref [95% CI]).

Usage: python mri_crps/tables.py [--tag main]
"""
import argparse
import json
from pathlib import Path

OUT = Path('/share/gpu0/jjwhit/crps_review/paper/mri')

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='main')
args = ap.parse_args()
R = json.load(open(OUT / f'results_{args.tag}.json'))
M = R['methods']
ref = R['ref']


def cell(e, j=None, nd=3, diff=False):
    g = (lambda v: v if j is None else v[j])
    if diff:
        if 'diff' not in e:
            return '—'
        return f"{g(e['diff']):+.{nd}f} [{g(e['diff_lo']):+.{nd}f}, {g(e['diff_hi']):+.{nd}f}]"
    return f"{g(e['val']):.{nd}f} [{g(e['lo']):.{nd}f}, {g(e['hi']):.{nd}f}]"


def table(rows, title):
    out = [f'**{title}**', '', '| metric | ' + ' | '.join(M) + ' | ' + ' | '.join(f'{k} - {ref}' for k in M if k != ref) + ' |',
           '|---' * (1 + len(M) + len(M) - 1) + '|']
    for name, d, j, nd in rows:
        out.append(f'| {name} | ' + ' | '.join(cell(d[k], j, nd) for k in M) + ' | '
                   + ' | '.join(cell(d[k], j, nd, diff=True) for k in M if k != ref) + ' |')
    return '\n'.join(out) + '\n'


rec = R['reconstruction']
rows = []
for P in [1, 2, 4, 8, 32]:
    rows.append((f'PSNR P={P}', rec[f'psnr_P{P}'], None, 2))
for P in [1, 8, 32]:
    rows.append((f'SSIM P={P}', rec[f'ssim_P{P}'], None, 4))
for P in [1, 8, 32]:
    rows.append((f'LPIPS P={P}', rec[f'lpips_P{P}'], None, 4))
rows.append(('APSD (32 samples)', rec['apsd32'], None, 5))
print(table(rows, f"Reconstruction on {R['n_slices']} test slices (P-sample mean; magnitude images)"))
print('| CFID (32 samples/slice) | ' + ' | '.join(f"{R['cfid'][k]['cfid']:.3f} [{R['cfid'][k]['lo']:.3f}, {R['cfid'][k]['hi']:.3f}]" for k in M)
      + ' | ' + ' | '.join(f"{R['cfid'][k]['diff']:+.3f} [{R['cfid'][k]['diff_lo']:+.3f}, {R['cfid'][k]['diff_hi']:+.3f}]" for k in M if k != ref) + ' |')
print('| CFID, rcGAN 72-slice test set | ' + ' | '.join(f"{R['cfid'][k]['cfid_rcgan72']:.3f}" for k in M) + ' |\n')
print('rcGAN 72-slice test set: ' + json.dumps(rec['rcgan_testset_72']) + '\n')

cal = R['marginal_calibration']
lv = R['exact_levels']
rows = [(f'pixel cov - {lv[j]:.3f}', cal['pix_cov_minus_exact'], j, 3) for j in range(len(lv))]
rows += [('pixel mean abs cov err', cal['pix_cov_abs_err_mean'], None, 3), ('pixel rank CE', cal['pix_rank_ce'], None, 3)]
rows += [(f'background-ring cov - {lv[j]:.3f}', cal['bg_cov_minus_exact'], j, 3) for j in [2, 5]]
print(table(rows, 'Per-pixel calibration (fg mask)'))
print('rank-histogram densities: ' + json.dumps(cal['pix_rankhist_density']) + '\n')

jc = R['joint_calibration']
rows = []
for s in [4, 8, 16, 32, 64]:
    rows += [(f'{s}px patch cov - {lv[j]:.3f}', jc[f'patch{s}_cov_minus_exact'], j, 3) for j in [2, 5]]
    rows += [(f'{s}px patch rank CE', jc[f'patch{s}_rank_ce'], None, 3),
             (f'{s}px patch v(1+1/N)/e', jc[f'patch{s}_var_corr_over_err2'], None, 3)]
rows += [(f'whole-fg mean cov - {lv[j]:.3f}', jc['global_cov_minus_exact'], j, 3) for j in [2, 5]]
rows += [('whole-fg mean rank CE', jc['global_rank_ce'], None, 3)]
rows += [('high-k sample/truth power', jc['highk_sample_over_truth'], None, 3),
         ('high-k post.-mean/truth power', jc['highk_postmean_over_truth'], None, 3),
         ('fair energy score', jc['es_fair'], None, 4), ('variogram score (local, p=0.5)', jc['vs_local_mean_over_lags'], None, 6)]
print(table(rows, 'Joint calibration'))
ke = jc['k_bins_edges_cycles_per_fov']
rows = [(f'k in [{ke[b]:.1f}, {ke[b + 1]:.1f})', jc['spec_sample_over_truth'], b, 3) for b in range(len(ke) - 1)]
print(table(rows, 'Per-sample / truth radially averaged power'))
rows = [(f'k in [{ke[b]:.1f}, {ke[b + 1]:.1f})', jc['spec_postmean_over_truth'], b, 3) for b in range(len(ke) - 1)]
print(table(rows, 'Posterior-mean / truth power'))
rows = [(f'k in [{ke[b]:.1f}, {ke[b + 1]:.1f})', jc['bandpower_rank_ce'], b, 3) for b in range(len(ke) - 1)]
print(table(rows, 'Band-power rank calibration error'))

dv = R['diversity']
rows = [('v(1+1/N)/e (pixels)', dv['spread_skill_v_corr_over_e'], None, 3),
        ('pairwise dist / truth dist', dv['pairdist_over_truthdist'], None, 3),
        ('frac pairs < half mean dist', dv['frac_pairs_closer_than_half_mean'], None, 4),
        ('min pair / mean pair', dv['min_pair_over_mean_pair'], None, 3)]
print(table(rows, 'Diversity / spread-skill'))
print('training: ' + json.dumps(R['training'], indent=1))
