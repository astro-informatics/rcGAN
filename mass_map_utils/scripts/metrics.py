import numpy as np
import json
import sys

sys.path.append("/home/jjwhit/rcGAN")
from data.lightning.MassMappingDataModule import MMDataTransform
from mass_map_utils.scripts.ks_utils import (
    backward_model,
    rmse,
    pearsoncoeff,
    psnr,
    snr,
)
from scipy import ndimage

data_dir = "/share/gpu0/jjwhit/samples/ks/rmse/"

mask = np.load(
    "/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_mask.npy", allow_pickle=True
).astype(bool)
std1 = np.load(
    "/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_std1.npy", allow_pickle=True
)
std2 = np.load(
    "/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_std2.npy", allow_pickle=True
)

kernel = MMDataTransform.compute_fourier_kernel(300)
np_kss = {}

r_ks = []
r_gan = []
rmse_ks = []
rmse_gan = []
psnr_ks = []
psnr_gan = []
snr_ks = []
snr_gan = []
r_std_abs_error = []
r_abs_error_sims = []
r_std_sims = []
all_psnr_vals = []
within_std_count = []

for map in range(1, 1001):
    np_gts = np.load(data_dir + f"np_gt_{map}.npy")
    np_samps = np.load(data_dir + f"np_samps_{map}.npy")
    np_avgs = np.load(data_dir + f"np_avgs_{map}.npy")
    np_stds = np.load(data_dir + f"np_stds_{map}.npy")

    gamma_sim = MMDataTransform.forward_model(np_gts, kernel) + (
        std1 * np.random.randn(300, 300) + 1.0j * std2 * np.random.randn(300, 300)
    )
    gamma_sim *= mask
    backward = backward_model(gamma_sim, kernel)
    np_kss = ndimage.gaussian_filter(backward, sigma=1 / 0.29)

    gt = np_gts.real
    ks = np_kss.real
    gan = np_avgs.real

    r_gan.append(pearsoncoeff(gt, gan, mask))
    r_ks.append(pearsoncoeff(gt, ks, mask))

    rmse_ks.append(rmse(ks, gt, mask))
    rmse_gan.append(rmse(gan, gt, mask))

    psnr_ks.append(psnr(gt, ks, mask))
    psnr_gan.append(psnr(gt, gan, mask))

    snr_ks.append(snr(gt, ks, mask))
    snr_gan.append(snr(gt, gan, mask))

    abs_error_gan = np.abs(gan - gt)

    relative_error = abs_error_gan.real / np_stds.real
    mean_relative_error = np.mean(relative_error[mask])
    std_relative_error = np.std(relative_error[mask])

    lower_bound = mean_relative_error - std_relative_error
    upper_bound = mean_relative_error + std_relative_error
    pix_within_std_relative = np.sum(
        (relative_error[mask] >= lower_bound) & (relative_error[mask] <= upper_bound)
    )
    total_masked_pix_relative = np.sum(mask)
    within_std_count_relative = pix_within_std_relative / total_masked_pix_relative
    within_std_count.append(within_std_count_relative)

    r_std_abs_error.append(pearsoncoeff(np_stds.real, abs_error_gan, mask))
    r_abs_error_sims.append(pearsoncoeff(gt, abs_error_gan, mask))
    r_std_sims.append(pearsoncoeff(np_stds.real, gt, mask))

    psnr_vals = []
    for n in range(1, 33):
        # Average the first `n` posterior samples to create a reconstruction
        recon = np.mean(np_samps[:n].real, axis=0)

        # Calculate PSNR for this reconstruction
        psnr_value = psnr(recon, np_gts.real, mask)
        psnr_vals.append(psnr_value)
    all_psnr_vals.append(psnr_vals)

results_dict = {
    "r_ks_avg": float(np.mean(r_ks)),
    "r_gan_avg": float(np.mean(r_gan)),
    "rmse_ks_avg": float(np.mean(rmse_ks)),
    "rmse_gan_avg": float(np.mean(rmse_gan)),
    "psnr_ks_avg": float(np.mean(psnr_ks)),
    "psnr_gan_avg": float(np.mean(psnr_gan)),
    "snr_ks_avg": float(np.mean(snr_ks)),
    "snr_gan_avg": float(np.mean(snr_gan)),
    "r_std_abs_error_avg": float(np.mean(r_std_abs_error)),
    "r_abs_error_sims_avg": float(np.mean(r_abs_error_sims)),
    "r_std_sims_avg": float(np.mean(r_std_sims)),
    "all_psnr_vals": np.array(all_psnr_vals).tolist(),
    "all_psnr_mean": np.mean(all_psnr_vals, axis=0).tolist(),
    "all_psnr_std": np.std(all_psnr_vals, axis=0).tolist(),
    "average_within_std_relative_count": float(np.mean(within_std_count)),
}

with open("results_dict", "w") as json_file:
    json.dump(results_dict, json_file)
