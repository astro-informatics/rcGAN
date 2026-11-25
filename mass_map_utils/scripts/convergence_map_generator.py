
import argparse
import datetime
import json
import os
import re
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Repo root (two levels up from mass_map_utils/scripts/), so the data module
# can be imported without a hardcoded absolute path.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Same forward model as training and normalisation.py (pulls in torch / lightning).
from data.lightning.MassMappingDataModule import MMDataTransform as MM

'''This script Builds a COSMOS-like dataset of weak lensing maps from kappaTNG weak lensing suites.

Each saved map is the central 300x300 crop of a 1024x1024 map, clipped at 0.7
(kappa values above 0.7 are set to 0.7) in all splits: train, val and test.

Output layout (root = cfg.data_path):
    root/train/kappa_00000.npy ... kappa_08499.npy   (8500 maps)
    root/val/kappa_00000.npy   ... kappa_00999.npy   (1000 maps)
    root/test/kappa_00000.npy  ... kappa_00499.npy   (500 maps)
'''

parser = argparse.ArgumentParser(description='Build the COSMOS-like kappaTNG convergence map dataset.')
parser.add_argument('--config', type=str, required=True, help='Path to the config file (e.g. ./configs/mass_map.yml)')
args = parser.parse_args()
with open(args.config, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = json.loads(json.dumps(cfg), object_hook=lambda d: types.SimpleNamespace(**d))

root = cfg.data_path
kappatng_path = cfg.kappatng_path
hist_path = os.path.join(cfg.cosmo_dir_path, 'hist_n_z.npy')

# All 100 x 100 kappaTNG (LP, run) pairs.
pairs = [(lp, run) for lp in range(1, 101) for run in range(1, 101)]
n_tng_slices = 41 # kappa00.dat ... kappa40.dat


def kappatng_file(lp, run, fname):
    """Path of the kappaTNG map read for redshift slice `fname` of one (LP, run) pair."""
    # Uses z=2.6 map (kappa40.dat, with appropriate weight) for redshifts beyond kappaTNG range.
    return os.path.join(kappatng_path, 'LP{:03d}'.format(lp), 'run{:03d}'.format(run),
                        'kappa{:02d}.dat'.format(min(fname, n_tng_slices - 1)))


# Check that every input exists before writing any output, so a missing file
# can never leave a partially written dataset behind.
required = [hist_path] + [kappatng_file(lp, run, k) for lp, run in pairs for k in range(n_tng_slices)]
missing = [p for p in required if not os.path.exists(p)]
if missing:
    max_listed = 20
    raise FileNotFoundError(
        f'{len(missing)} of {len(required)} required input files are missing'
        + (f' (first {max_listed} listed)' if len(missing) > max_listed else '') + ':\n'
        + '\n'.join(missing[:max_listed])
    )


# Normalisation stats written back to the --config after all maps are saved.
# Check the keys now (and again at write time) so a missing key fails before
# hours of generation rather than after.
STAT_KEYS = ('kappa_mean', 'kappa_std', 'shear_mag_mean', 'shear_mag_std')
SCRIPT_NAME = 'convergence_map_generator.py'


def stat_key_pattern(key):
    """Matches only the ACTIVE `key:` line (column 0), never a commented history line."""
    return re.compile(rf'^{re.escape(key)}:(.*)$', flags=re.M)


def yaml_float(val):
    """Full-precision float literal that PyYAML reads back as a float.

    repr() gives e.g. '5e-05' for small values, which PyYAML (YAML 1.1) loads
    as a *string*; force a '.' in the mantissa ('5.0e-05').
    """
    val = float(val)
    if not np.isfinite(val):
        raise ValueError(f'Refusing to write non-finite stat {val!r} to the config.')
    s = repr(val)
    mantissa, e, exponent = s.partition('e')
    if e and '.' not in mantissa:
        s = f'{mantissa}.0e{exponent}'
    return s


def replace_stat_line(cfg_text, key, val, note, script_name, today):
    """Replace the active `key:` line with a commented history line plus the new value.

    The active line becomes two lines:
        # <key>: <old value> # old value, replaced by <script_name> on <today>
        <key>: <new value> # <note>
    The old line's trailing comment is dropped. Only an uncommented line at
    column 0 is matched, so repeated runs stack history lines and never touch
    earlier commented ones. Every other line is preserved.
    """
    matches = list(stat_key_pattern(key).finditer(cfg_text))
    if not matches:
        raise KeyError(f'{key} not found in {args.config}')
    if len(matches) > 1:
        raise ValueError(f'{key} is set {len(matches)} times in {args.config}; keep only one active line.')
    m = matches[0]
    old_val = re.split(r'\s#', m.group(1), maxsplit=1)[0].strip()
    new_lines = (
        f'# {key}: {old_val} # old value, replaced by {script_name} on {today}\n'
        f'{key}: {yaml_float(val)} # {note}'
    )
    # Slice rather than re.sub so backslashes in the old value are kept literally.
    return cfg_text[:m.start()] + new_lines + cfg_text[m.end():]


def require_int_cfg(name, minimum):
    """Return integer cfg.<name>, raising a clear error if it is missing, not an int or < minimum."""
    if not hasattr(cfg, name):
        raise KeyError(f'{name} not found in {args.config}; add it (see configs/mass_map.yml).')
    val = getattr(cfg, name)
    if isinstance(val, bool) or not isinstance(val, int):
        raise TypeError(f'{name} in {args.config} must be an integer, got {val!r}.')
    if val < minimum:
        raise ValueError(f'{name} in {args.config} must be >= {minimum}, got {val}.')
    return val


with open(args.config, 'r') as f:
    _cfg_text = f.read()
_missing_keys = [k for k in STAT_KEYS if not stat_key_pattern(k).search(_cfg_text)]
if _missing_keys:
    raise KeyError(f'{_missing_keys} not found in {args.config}; add them (any value) before generating.')
_dup_keys = [k for k in STAT_KEYS if len(stat_key_pattern(k).findall(_cfg_text)) > 1]
if _dup_keys:
    raise ValueError(f'{_dup_keys} are set more than once in {args.config}; keep only one active line each.')

# Shear-noise settings for the shear_mag_* stats (same keys as normalisation.py).
shear_noise_seed = require_int_cfg('shear_noise_seed', 0) # default_rng rejects negative seeds
shear_noise_draws = require_int_cfg('shear_noise_draws', 1)

# COSMOS noise std maps for the shear-magnitude stats (as in normalisation.py).
# Loaded before any output is written, so a missing file aborts cleanly.
std1 = np.load(os.path.join(cfg.cosmo_dir_path, 'cosmos_std1.npy'), allow_pickle=True)
std2 = np.load(os.path.join(cfg.cosmo_dir_path, 'cosmos_std2.npy'), allow_pickle=True)


# Configures output directories. One explicit, distinct directory per split.
# Refuse to write into a non-empty split dir so stale files from an earlier run
# can never be mixed into the new dataset.
split_dirs = {
    'train': os.path.join(root, 'train'),
    'val': os.path.join(root, 'val'),
    'test': os.path.join(root, 'test'),
}

os.makedirs(root, exist_ok=True)
for split, split_dir in split_dirs.items():
    os.makedirs(split_dir, exist_ok=True)
    if os.listdir(split_dir):
        raise RuntimeError(f'{split} directory {split_dir} is not empty; generate into a fresh data_path.')


# Configures redshift distribution.
redshift_distribution = np.load(hist_path, allow_pickle=True)

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
order = np.random.default_rng(seed).permutation(len(pairs))

n_train, n_val, n_test = 8500, 1000, 500
assert n_train + n_val + n_test == len(pairs) == 10000
split_order = {
    'train': order[:n_train],
    'val': order[n_train:n_train + n_val],
    'test': order[n_train + n_val:],
}

# Cropping and clipping are folded in from cropper.py (which is now no longer
# needed in the pipeline): only the central 300x300 crop of each map is written,
# never the full 1024x1024 map.
# Every map in every split (train, val and test) is clipped at 0.7
# (equivalent to cropper.py's in-place kappa[kappa>0.7]=0.7).
center_size = 300 # Size of COSMOS maps
center_start = (1024 - center_size) // 2
center_end = center_start + center_size

# Normalisation stats are accumulated on the fly over the train split only,
# from the same cropped + clipped maps that are saved (see RunningStats).
for name, arr in (('cosmos_std1', std1), ('cosmos_std2', std2)):
    if arr.shape != (center_size, center_size):
        raise ValueError(f'{name}.npy has shape {arr.shape}, expected {(center_size, center_size)}.')
kernel = MM.compute_fourier_kernel(center_size) # computed once
# Dedicated RNG for the shear noise (cfg.shear_noise_seed), separate from the
# split permutation's default_rng(0) above, so the split is unchanged and the
# shear stats are reproducible.
shear_noise_rng = np.random.default_rng(shear_noise_seed)


class RunningStats:
    """Population mean / std over every pixel of many maps.

    Each update (one map, or one noise draw of a map) contributes
    (count, mean, M2) computed in float64, merged with Chan et al.'s parallel
    update, so there is no sum-of-squares cancellation.
    Matches normalisation.py's definition:
        mean = mean over all pixels
        std  = sqrt(sum((x - mean)^2) / (n_updates * im_size^2))
    """

    def __init__(self):
        self.n_updates = 0
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        n_b = x.size
        mean_b = x.mean()
        m2_b = np.sum((x - mean_b) ** 2)
        n = self.count + n_b
        delta = mean_b - self.mean
        self.mean += delta * n_b / n
        self.m2 += m2_b + delta ** 2 * self.count * n_b / n
        self.count = n
        self.n_updates += 1

    @property
    def std(self):
        return np.sqrt(self.m2 / self.count)


kappa_stats = RunningStats()
shear_mag_stats = RunningStats()


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
    kappa_tot = np.zeros((1024, 1024))
    for fname in range(len(redshift_vals_array)):
        with open(kappatng_file(lp, run, fname), 'rb') as f:
            dummy = np.fromfile(f, dtype="int32", count=1)
            kappa = np.fromfile(f, dtype="float", count=1024*1024)
            dummy = np.fromfile(f, dtype="int32", count=1)

        kappa = kappa.reshape((1024, 1024))
        kappa_tot += (omega[fname]/norm) * kappa
    return kappa_tot


for split in ('train', 'val', 'test'):
    split_dir = split_dirs[split]
    for index, pair_idx in enumerate(split_order[split]):
        lp, run = pairs[pair_idx]
        kappa_tot = build_map(lp, run)

        # Crop to the central 300x300 region in memory, then clip at 0.7.
        kappa_crop = kappa_tot[center_start:center_end, center_start:center_end]
        kappa_crop = np.minimum(kappa_crop, 0.7)

        if split == 'train':
            # Normalisation stats from the train split only. The shear is not
            # stored anywhere, so it is simulated here exactly as in
            # normalisation.py: noiseless shear from the clipped 300x300 kappa,
            # plus COSMOS noise, then its magnitude.
            kappa_stats.update(kappa_crop)
            # Noise-free shear once per map; then shear_noise_draws independent
            # noise draws, each adding one more batch of pixels to the pooled
            # stats. Never average noise / noisy shear across draws before
            # abs() or accumulating: that would shrink the noise variance.
            shear = MM.forward_model(kappa_crop, kernel)
            for _ in range(shear_noise_draws):
                noisy_shear = shear + (
                    std1 * shear_noise_rng.standard_normal((center_size, center_size))
                    + 1j * std2 * shear_noise_rng.standard_normal((center_size, center_size))
                )
                shear_mag_stats.update(np.abs(noisy_shear))

        save_path = os.path.join(split_dir, 'kappa_{:05d}.npy'.format(index))
        np.save(save_path, kappa_crop, allow_pickle=True)


# All maps saved: write the train-split stats back into the --config, editing
# only the four stat lines (old values kept as commented history lines) so all
# other lines and comments are preserved.
if kappa_stats.n_updates != n_train or shear_mag_stats.n_updates != n_train * shear_noise_draws:
    raise RuntimeError(
        f'Stats cover {kappa_stats.n_updates} kappa / {shear_mag_stats.n_updates} shear updates, '
        f'expected {n_train} / {n_train * shear_noise_draws} ({n_train} maps x {shear_noise_draws} draws).'
    )
stats = {
    'kappa_mean': kappa_stats.mean,
    'kappa_std': kappa_stats.std,
    'shear_mag_mean': shear_mag_stats.mean,
    'shear_mag_std': shear_mag_stats.std,
}
for key, val in stats.items():
    print(f'{key}: {val!r}')

today = datetime.date.today().isoformat()
with open(args.config, 'r') as f:
    cfg_text = f.read()
for key, val in stats.items():
    note = f'computed by {SCRIPT_NAME} from {n_train} train maps'
    if key.startswith('shear_mag'):
        note += f' (noise seed {shear_noise_seed}, {shear_noise_draws} draws)'
    cfg_text = replace_stat_line(cfg_text, key, val, note, SCRIPT_NAME, today)
with open(args.config, 'w') as f:
    f.write(cfg_text)
print('Wrote', ', '.join(stats), 'to', args.config)
