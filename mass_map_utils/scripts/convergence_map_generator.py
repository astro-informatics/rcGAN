
import argparse
import csv
import json
import os
import types

import matplotlib.pyplot as plt
import numpy as np
import yaml

'''This script Builds a COSMOS-like dataset of weak lensing maps from kappaTNG weak lensing suites.

Output layout (root = cfg.data_path):
    root/train/kappa_00000.npy ... kappa_08499.npy   (8500 maps)
    root/val/kappa_00000.npy   ... kappa_00499.npy   (500 maps)
    root/test/kappa_00000.npy  ... kappa_00999.npy   (1000 maps)
    root/manifest.csv          (split,index,lp,run,clipped,seed)
'''

DEFAULT_KAPPATNG_PATH = '/share/gpu0/jjwhit/kappaTNG_suites'

parser = argparse.ArgumentParser(description='Build the COSMOS-like kappaTNG convergence map dataset.')
parser.add_argument('--config', type=str, required=True, help='Path to the config file (e.g. ./configs/mass_map.yml)')
args = parser.parse_args()
with open(args.config, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = json.loads(json.dumps(cfg), object_hook=lambda d: types.SimpleNamespace(**d))

root = cfg.data_path
kappatng_path = getattr(cfg, 'kappatng_path', DEFAULT_KAPPATNG_PATH)


# Configures output directories. One explicit, distinct directory per split.
# Refuse to write into a non-empty split dir (or over an existing manifest) so
# stale files from an earlier run can never be mixed into the new dataset.
split_dirs = {
    'train': os.path.join(root, 'train'),
    'val': os.path.join(root, 'val'),
    'test': os.path.join(root, 'test'),
}
manifest_path = os.path.join(root, 'manifest.csv')

os.makedirs(root, exist_ok=True)
for split, split_dir in split_dirs.items():
    os.makedirs(split_dir, exist_ok=True)
    if os.listdir(split_dir):
        raise RuntimeError(f'{split} directory {split_dir} is not empty; generate into a fresh data_path.')
if os.path.exists(manifest_path):
    raise RuntimeError(f'{manifest_path} already exists; generate into a fresh data_path.')


# Configures redshift distribution.
redshift_distribution = np.load(os.path.join(cfg.cosmo_dir_path, 'hist_n_z.npy'), allow_pickle=True)

redshift_vals_array = np.array((0,
                                0.034, 0.070, 0.105, 0.142, 0.179,
                                0.216, 0.255, 0.294, 0.335, 0.376,
                                0.418, 0.462, 0.506, 0.552, 0.599,
                                0.648, 0.698, 0.749, 0.803, 0.858,
                                0.914, 0.973, 1.034, 1.097, 1.163,
                                1.231, 1.302, 1.375, 1.452, 1.532,
                                1.615, 1.703, 1.794, 1.889, 1.989,
                                2.094, 2.203, 2.319, 2.440, 2.568,
                                2.605, 2.642, 2.679,
                                2.716, 2.755, 2.794, 2.835, 2.876,
                                2.918, 2.962, 3.006, 3.052, 3.099,
                                3.148, 3.198, 3.249, 3.303, 3.358,
                                3.414, 3.473, 3.534, 3.597, 3.663,
                                3.731, 3.802, 3.875, 3.952, 4.032,
                                4.115, 4.203, 4.294, 4.389, 4.489,
                                4.594, 4.703, 4.819, 4.940, 5.068)
                                )

z = np.arange(0, 5.5, .01)

bin_edges = (np.concatenate([[0],
            [(redshift_vals_array[i] - redshift_vals_array[i-1])/2 +
            redshift_vals_array[i-1] for i in range(1, len(redshift_vals_array))],
            [redshift_vals_array[-1] + (redshift_vals_array[-1] - redshift_vals_array[-2])/2]]))
# Create a histogram using your specified bin edges
hist, bin_edges, _ = (plt.hist(z, weights=redshift_distribution, bins=bin_edges, color="purple",
                        alpha=0.5, ec="k", histtype='stepfilled', density=True, label='Normalized'))
plt.legend()
plt.savefig('Redshift Distribution Histogram')
pdf = hist


# Configures kappaTNG files.
# One seeded permutation of all (LP, run) pairs, so every pair is used exactly
# once. (Shuffling the LP and run columns independently samples pairs with
# replacement and produced duplicate maps across train/test/val.)
# Note: the split is by (LP, run) pair, not by LP, so maps that share an LP can
# appear in different splits.
seed = 0
pairs = [(lp, run) for lp in range(1, 101) for run in range(1, 101)]
order = np.random.default_rng(seed).permutation(len(pairs))

n_train, n_val, n_test = 8500, 500, 1000
assert n_train + n_val + n_test == len(pairs)
split_order = {
    'train': order[:n_train],
    'val': order[n_train:n_train + n_val],
    'test': order[n_train + n_val:],
}

# Cropping and clipping are folded in from cropper.py (which is now no longer
# needed in the pipeline): only the central 300x300 crop of each map is written,
# never the full 1024x1024 map.
# Train and val maps are clipped at 0.7 (equivalent to cropper.py's in-place
# kappa[kappa>0.7]=0.7); test maps are deliberately NOT clipped, matching cropper.py.
center_size = 300 # Size of COSMOS maps
center_start = (1024 - center_size) // 2
center_end = center_start + center_size
clip_value = 0.7
clip_split = {'train': True, 'val': True, 'test': False}


# Precompute redshift weights once (they do not depend on LP or run).
# Bin size halved for first and last bins.
# delta_z is the range of each bin.
omega = np.zeros(len(redshift_vals_array))
for i in range(len(redshift_vals_array)):
    if i == 0:
        delta_z = (redshift_vals_array[i + 1] - redshift_vals_array[i])/2

    elif i == len(redshift_vals_array) - 1:
        delta_z = (redshift_vals_array[i] - redshift_vals_array[i - 1])/2

    else:
        delta_z = (
            (redshift_vals_array[i + 1] - redshift_vals_array[i]) /2
            + (redshift_vals_array[i] - redshift_vals_array[i - 1]) /2
        )

    omega[i] = pdf[i] * delta_z
norm = omega.sum()


def build_map(lp, run):
    """Returns the redshift-weighted 1024x1024 convergence map for one (LP, run) pair."""
    path_to_load = os.path.join(kappatng_path, 'LP{:03d}'.format(lp), 'run{:03d}'.format(run))
    kappa_tot = np.zeros((1024, 1024))
    for fname in range(len(redshift_vals_array)):
        if fname in range(0, 41):
            full_path = os.path.join(path_to_load, 'kappa' + str(fname).zfill(2) + '.dat')
        else:
            # Uses z=2.6 map (with appropriate weight) for redshifts beyond kappaTNG range.
            full_path = os.path.join(path_to_load, 'kappa40.dat')

        if not os.path.exists(full_path):
            raise FileNotFoundError(f'The file at {full_path} does not exist.')

        with open(full_path, 'rb') as f:
            dummy = np.fromfile(f, dtype="int32", count=1)
            kappa = np.fromfile(f, dtype="float", count=1024*1024)
            dummy = np.fromfile(f, dtype="int32", count=1)

        kappa = kappa.reshape((1024, 1024))
        kappa_tot += (omega[fname]/norm) * kappa
    return kappa_tot


manifest_rows = []
for split in ('train', 'val', 'test'):
    split_dir = split_dirs[split]
    clipped = clip_split[split]
    for index, pair_idx in enumerate(split_order[split]):
        lp, run = pairs[pair_idx]
        kappa_tot = build_map(lp, run)

        # Crop to the central 300x300 region in memory.
        kappa_crop = kappa_tot[center_start:center_end, center_start:center_end]
        if clipped:
            kappa_crop = np.minimum(kappa_crop, clip_value)

        save_path = os.path.join(split_dir, 'kappa_{:05d}.npy'.format(index))
        np.save(save_path, kappa_crop, allow_pickle=True)
        manifest_rows.append((split, index, lp, run, clipped, seed))


with open(manifest_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(('split', 'index', 'lp', 'run', 'clipped', 'seed'))
    writer.writerows(manifest_rows)
