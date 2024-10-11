import glob
import os
import numpy as np


# kappa20 - z = 0.858

# Use this for the raw, .dat files
# src_path = "/disk/xray99/jdm/kappaTNG/kappaTNG-Hydro/LP*/run*/kappa20.dat"

# Use this is files have already been converted to .npy files
src_path = "/share/gpu0/jjwhit/mass_map_dataset/kappa_dataset/*.npy"
all_files = glob.glob(src_path)

# Define the destination folder
dst_path = "/share/gpu0/jjwhit/mass_map_dataset/kappa20_cropped/"


if not os.path.exists(dst_path):
    # Create a new directory because it does not exist
    os.makedirs(dst_path)
    print("The new directory has been made!")

dst_train_path = dst_path + "kappa_train/"
dst_test_path = dst_path + "kappa_test/"
dst_val_path = dst_path + "kappa_val/"

if not os.path.exists(dst_train_path):
    os.makedirs(dst_train_path)
if not os.path.exists(dst_test_path):
    os.makedirs(dst_test_path)
if not os.path.exists(dst_val_path):
    os.makedirs(dst_val_path)


img_number = 1
total_nb_files = len(all_files)
ng = 1024

# Set seed
np.random.seed(0)
# Shuffle files
np.random.shuffle(all_files)


# Includes the transformation from .dat to .npy

for fname in all_files:
    print("Processing file n", img_number)
    with open(fname, "rb") as f:
        dummy = np.fromfile(f, dtype="int32", count=1)
        kappa = np.fromfile(f, dtype="float", count=ng * ng)
        dummy = np.fromfile(f, dtype="int32", count=1)

        kappa = kappa.reshape((ng, ng))
        kappa = f
        kappa[kappa > 0.7] = 0.7

        center_size = 384  # Size of MRI images
        center_start = (ng - center_size) // 2
        center_end = center_start + center_size

        kappa_cropped = kappa[center_start:center_end, center_start:center_end]

        # Using 85% of data for training
        if (img_number / total_nb_files) <= 0.85:
            dst_dir = dst_train_path
            # Using 10% of data for testing
        elif (img_number / total_nb_files) > 0.85 and (
            img_number / total_nb_files
        ) <= 0.95:
            dst_dir = dst_test_path
            # Using 5% of data for validation
        else:
            dst_dir = dst_val_path

        save_path = "{:s}{:s}{:05d}{:s}".format(
            dst_dir, "kappa_run_", img_number, ".npy"
        )

        np.save(save_path, kappa_cropped, allow_pickle=True)

        img_number += 1

# Takes kappa.npy files and crops them
# for fname in all_files:
#     print('Processing file n', img_number)
#     kappa = np.load(fname, allow_pickle=True)
#     kappa[kappa>0.7]=0.7

#     center_size = 384  #Size of MRI images
#     center_start = (1024 - center_size) // 2
#     center_end =  center_start + center_size

#     kappa_cropped = kappa[center_start:center_end,  center_start:center_end]
#             # Using 85% of data for training
#     if (img_number/total_nb_files) <= 0.85:
#         dst_dir = dst_train_path
#         # Using 10% of data for testing
#     elif (img_number/total_nb_files) > 0.85 and (img_number/total_nb_files) <= 0.95:
#         dst_dir = dst_test_path
#         # Using 5% of data for validation
#     else:
#         dst_dir = dst_val_path

#     save_path = '{:s}{:s}{:05d}{:s}'.format(dst_dir, "kappa_run_", img_number, ".npy")

#     np.save(save_path, kappa_cropped, allow_pickle=True)

#     img_number +=1
