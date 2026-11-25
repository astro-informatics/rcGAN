import numpy as np
import os
import glob
import sys
sys.path.append("/home/jjwhit/rcGAN")
import yaml
import json
import types
import re
import datetime
from utils.parse_args import create_arg_parser
from data.lightning.MassMappingDataModule import MMDataTransform as MM

SCRIPT_NAME = 'normalisation.py'


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
    (Same function as in convergence_map_generator.py.)
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


def load_object(dct):
    return types.SimpleNamespace(**dct)
args = create_arg_parser().parse_args()
with open(args.config, "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = json.loads(json.dumps(cfg), object_hook=load_object)

# Shear-noise settings for the shear_mag_* stats (same keys as the generator).
shear_noise_seed = require_int_cfg('shear_noise_seed', 0) # default_rng rejects negative seeds
shear_noise_draws = require_int_cfg('shear_noise_draws', 1)

data_dir = os.path.join(cfg.data_path, 'train')

# Sorted order is the generator's seeded random permutation, so the first 500
# files are a random sample of the training set.
files = sorted(glob.glob(os.path.join(data_dir, 'kappa_*.npy')))
kappa_mean=[]
shear_mean=[]

im_size=cfg.im_size

mask =  np.load(
    cfg.cosmo_dir_path + 'cosmos_mask.npy', allow_pickle=True
).astype(bool)
std1 = np.load(
    cfg.cosmo_dir_path + 'cosmos_std1.npy', allow_pickle=True
)
std2 = np.load(
    cfg.cosmo_dir_path + 'cosmos_std2.npy', allow_pickle=True
)
kernel = MM.compute_fourier_kernel(im_size)

# Shear noise uses its own Generator seeded with cfg.shear_noise_seed (not the
# global np.random state). A single Generator would give the two passes below
# different noise, so it is reset to the same seed at the start of each pass:
# the mean and std passes then see identical noise. Each map gets
# shear_noise_draws independent draws, pooled as extra pixels in the stats
# (never averaged across draws before abs(), which would shrink the noise).

# Pass 1: means.
shear_noise_rng = np.random.default_rng(shear_noise_seed)
count = 0
for file in files:
    data = np.load(file)
    kappa_mean.append(np.mean(data))
    shear = MM.forward_model(data, kernel) # noise-free shear, once per map
    for _ in range(shear_noise_draws):
        noisy_shear = shear + (
                std1 * shear_noise_rng.standard_normal((im_size, im_size))
                + 1.j * std2 * shear_noise_rng.standard_normal((im_size, im_size))
        )
        #Decompose shear into magnitude and phase
        magnitude = np.abs(noisy_shear)
        #Compute mean of magnitude (every draw has im_size^2 pixels, so the
        #mean of these per-draw means is the pooled mean)
        shear_mean.append(np.mean(magnitude))
    count += 1
    if count >= 500:
        break

total_kappa_mean = np.mean(kappa_mean)
print('Total kappa mean is: ', total_kappa_mean)
total_shear_mean = np.mean(shear_mean)
print('Total shear magnitude mean is: ', total_shear_mean)

# Pass 2: stds (same files, same order, rng reset -> same noise as pass 1).
shear_noise_rng = np.random.default_rng(shear_noise_seed)
count=0
std_i = 0
std_k = 0
for file in files:
    data = np.load(file)
    kappa_i = (data - total_kappa_mean)**2
    std_i += np.sum(kappa_i)
    shear = MM.forward_model(data, kernel) # noise-free shear, once per map
    for _ in range(shear_noise_draws):
        noisy_shear = shear + (
                std1 * shear_noise_rng.standard_normal((im_size, im_size))
                + 1.j * std2 * shear_noise_rng.standard_normal((im_size, im_size))
        )
        #Decompose shear into magnitude and phase
        magnitude = np.abs(noisy_shear)
        shear_i = (magnitude - total_shear_mean)**2
        std_k += np.sum(shear_i)
    count += 1
    if count >= 500:
        break

total_std_kappa = np.sqrt(std_i/(count*im_size**2))
print('Kappa standard deviation is: ', total_std_kappa)
total_std_shear = np.sqrt(std_k/(count*shear_noise_draws*im_size**2))
print('Shear mean standard deviation is: ', total_std_shear)
# Write kappa_mean / kappa_std / shear_mag_mean / shear_mag_std back into the
# config, editing only those four lines so the rest of the file (comments,
# ordering) is preserved; each old value is kept as a commented history line.
# Standalone fallback: convergence_map_generator.py writes the same keys from
# all train maps; this script uses the first 500.
today = datetime.date.today().isoformat()
with open(args.config, "r") as f:
    cfg_text = f.read()
for key, val in (
    ('kappa_mean', total_kappa_mean),
    ('kappa_std', total_std_kappa),
    ('shear_mag_mean', total_shear_mean),
    ('shear_mag_std', total_std_shear),
):
    note = f'computed by {SCRIPT_NAME} from {count} train maps'
    if key.startswith('shear_mag'):
        note += f' (noise seed {shear_noise_seed}, {shear_noise_draws} draws)'
    cfg_text = replace_stat_line(cfg_text, key, val, note, SCRIPT_NAME, today)
with open(args.config, "w") as f:
    f.write(cfg_text)
print('Wrote kappa_mean, kappa_std, shear_mag_mean and shear_mag_std to', args.config)
