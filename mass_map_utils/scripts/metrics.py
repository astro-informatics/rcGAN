import numpy as np
import json
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/jjwhit/rcGAN")
from data.lightning.MassMappingDataModule import MMDataTransform
from mass_map_utils.scripts.ks_utils import (
    backward_model,
    rmse,
    pearsoncoeff,
    psnr,
)

data_dir = "/share/gpu0/jjwhit/samples/test_set/"

mask = np.load(
    "/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_mask.npy", allow_pickle=True
).astype(bool)
std1 = np.load(
    "/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_std1.npy", allow_pickle=True
)
std2 = np.load(
    "/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_std2.npy", allow_pickle=True
)

def ecp(samples, gt, mask, level, lam=1.0):
    """
    Computes empirical coverage probability (for the unmasked pixels only).

    Args:
        samples (np.ndarray): The samples used to create a reconstruction. Shape [N, H, W]
        gt (np.ndarray): Ground truth map
        mask (np.ndarray): Survey mask. Array of boolean values.
        level (float): Credibility level
        lam (float): RCPS calibration scaling factor

    Returns:    
        ecp (float): fraction of unmasked ground truth pixels within the credible interval
    """
    lower_q = 100*(1-level)/2
    upper_q = (100 - lower_q)
    lower = np.percentile(samples, lower_q, axis=0)
    upper = np.percentile(samples, upper_q, axis=0)
    inside_interval = (gt >= lower) & (gt <= upper)
    inside_interval = inside_interval[mask==1]
    return np.mean(inside_interval)

kernel = MMDataTransform.compute_fourier_kernel(300)

r_ks = []
r_gan = []
rmse_ks = []
rmse_gan = []
psnr_ks = []
psnr_gan = []
all_psnr_vals = []
all_pearson_vals = []
ecp_vals = {q: [] for q in range(5, 100, 5)}



# for map in range(1, 1001):
for map in range(1, 3):
    np_gts = np.load(data_dir + f"kappa/np_gt_{map:04d}.npy")
    np_samps = np.load(data_dir + f"recon/np_samps_{map:04d}.npy")
    np_gamma = np.load(data_dir + f"gamma/np_gamma_{map:04d}.npy")

    gamma_sim = mask * np_gamma
    backward = backward_model(gamma_sim, kernel)
    # np_kss = ndimage.gaussian_filter(backward, sigma=1 / 0.29)
    np_kss = np.flipud(ndimage.rotate(ndimage.gaussian_filter(backward, sigma=1/0.29), 270))

    gt = np_gts.real
    ks = np_kss.real
    gan = np.mean(np_samps, axis=0).real

    r_gan.append(pearsoncoeff(gt, gan, mask))
    r_ks.append(pearsoncoeff(gt, ks, mask))

    rmse_ks.append(rmse(ks, gt, mask))
    rmse_gan.append(rmse(gan, gt, mask))

    psnr_ks.append(psnr(gt, ks, mask))
    psnr_gan.append(psnr(gt, gan, mask))
    abs_error_gan = np.abs(gan - gt)
    psnr_vals = []
    pearson_vals = []
    for n in range(1, 33):
        # Average the first `n` posterior samples to create a reconstruction
        recon = np.mean(np_samps[:n].real, axis=0)

        # Calculate PSNR for this reconstruction
        psnr_value = psnr(recon, np_gts.real, mask)
        psnr_vals.append(psnr_value)
        pearson_value = pearsoncoeff(recon, np_gts.real, mask)
        pearson_vals.append(pearson_value)
    all_psnr_vals.append(psnr_vals)
    all_pearson_vals.append(pearson_vals)

    for q in ecp_vals.keys():
        level = q/100
        ecp_vals[q].append(ecp(np_samps.real, gt, mask, level=level))

results_dict = {
    "r_ks_avg": float(np.mean(r_ks)),
    "r_gan_avg": float(np.mean(r_gan)),
    "rmse_ks_avg": float(np.mean(rmse_ks)),
    "rmse_gan_avg": float(np.mean(rmse_gan)),
    "psnr_ks_avg": float(np.mean(psnr_ks)),
    "psnr_gan_avg": float(np.mean(psnr_gan)),
    "all_psnr_vals": np.array(all_psnr_vals).tolist(),
    "all_psnr_mean": np.mean(all_psnr_vals, axis=0).tolist(),
    "all_psnr_std": np.std(all_psnr_vals, axis=0).tolist(),
    "all_pearson_vals": np.array(all_pearson_vals).tolist(),
    "all_pearson_mean": np.mean(all_pearson_vals, axis=0).tolist(),
    "all_pearson_std": np.std(all_pearson_vals, axis=0).tolist(),
    "ecp_vals": {q: float(np.mean(ecp_vals[q])) for q in ecp_vals.keys()}
}

with open("results_dict_new", "w") as json_file:
    json.dump(results_dict, json_file)

quantiles = list(ecp_vals.keys())
ecp_means = [np.mean(ecp_vals[q]) for q in quantiles]

plt.figure(figsize=(8, 6))
plt.plot(quantiles, ecp_means, marker='o', linestyle='-')
plt.plot([0, 100], [0, 1], "k--", label="Ideal")  # Reference line for perfect calibration
plt.xlabel("Credible Interval (%)")
plt.ylabel("Empirical Coverage Probability")
plt.title("ECP Plot")
plt.legend()
plt.grid()
plt.savefig("/home/jjwhit/rcGAN/figures/ecp_plot_3.png")