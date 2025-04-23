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

data_dir = "/share/gpu0/jjwhit/samples/real_output/"

mask = np.load(
    "/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_mask.npy", allow_pickle=True
).astype(bool)
std1 = np.load(
    "/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_std1.npy", allow_pickle=True
)
std2 = np.load(
    "/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_std2.npy", allow_pickle=True
)

def ecp(samples, gt, lower_quant, upper_quant, mask):
    samples = samples[mask==1]
    gt = gt[mask==1]
    lower = np.percentile(samples, lower_quant)
    upper = np.percentile(samples, upper_quant)
    inside_interval = (gt >= lower) & (gt <= upper)
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
        lower_quant = (100 - q) / 2
        upper_quant = 100 - lower_quant
        ecp_vals[q].append(ecp(np_avgs, gt, lower_quant, upper_quant, mask))

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

with open("results_dict_real_output", "w") as json_file:
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
plt.savefig("/home/jjwhit/rcGAN/figures/ecp_plot.png")