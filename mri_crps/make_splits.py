"""Fixed train/val/test volume lists for the MRI CRPS experiments (from manifest/scan.json).

Policy (rcGAN paper / repo): T2 brain (AXT2) multicoil volumes with > 8 coils, top slices only.
  train: multicoil_train, kspace width >= 384; shuffled with seed 0; runs use a prefix of N_TRAIN volumes.
  val:   multicoil_val, width > 384, minus the 5 files the repo excludes; shuffled with seed 1000;
         first N_VAL volumes, top 8 slices (beta tuning / validation PSNR).
  test:  the 12 rcGAN test volumes (small_T2_test) + the next N_TEST_EXTRA val volumes, top 6 slices
         (6 slices as in the repo's test set).
"""
import json
import random

OUT = '/share/gpu0/jjwhit/crps_review/paper/mri/manifest/'
N_TRAIN_MAX, N_VAL, N_TEST_EXTRA = 1528, 24, 24
EXCLUDE = {'file_brain_AXT2_209_2090296.h5', 'file_brain_AXT2_200_2000250.h5', 'file_brain_AXT2_201_2010106.h5',
           'file_brain_AXT2_204_2130024.h5', 'file_brain_AXT2_210_2100025.h5'}

scan = json.load(open(OUT + 'scan.json'))
train = sorted(r['name'] for r in scan['multicoil_train'] if r['shape'][3] >= 384 and r['shape'][1] > 8)
val = sorted(r['name'] for r in scan['multicoil_val']
             if r['shape'][3] > 384 and r['shape'][1] > 8 and r['name'] not in EXCLUDE)
random.Random(0).shuffle(train)
random.Random(1000).shuffle(val)
splits = dict(
    train=[dict(dir='multicoil_train', name=n, n_slices=8) for n in train[:N_TRAIN_MAX]],
    val=[dict(dir='multicoil_val', name=n, n_slices=8) for n in val[:N_VAL]],
    test=[dict(dir='small_T2_test', name=r['name'], n_slices=6) for r in scan['small_T2_test']]
    + [dict(dir='multicoil_val', name=n, n_slices=6) for n in val[N_VAL:N_VAL + N_TEST_EXTRA]],
)
json.dump(splits, open(OUT + 'splits.json', 'w'), indent=1)
print({k: (len(v), sum(e['n_slices'] for e in v)) for k, v in splits.items()})
