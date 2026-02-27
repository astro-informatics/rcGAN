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

# data_dir = "/share/gpu0/jjwhit/samples/hpd/"
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

def mse(a, b):
    return (
        np.linalg.norm(a - b, axis=(-2, -1), ord='fro')
    ) ** 2

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
mse_coverage = {q: 0 for q in range(5, 100, 5)}

# num_imgs = 790
num_imgs = 1000

for i in range(1, num_imgs + 1):
    np_gts = np.load(data_dir + f"kappa/np_gt_{i:04d}.npy")
    np_samps = np.load(data_dir + f"recon/np_samps_{i:04d}.npy")
    np_gamma = np.load(data_dir + f"gamma/np_gamma_{i:04d}.npy")

    gamma_sim = mask * np_gamma
    backward = backward_model(gamma_sim, kernel)
    # np_kss = ndimage.gaussian_filter(backward, sigma=1 / 0.29)
    np_kss = np.flipud(ndimage.rotate(ndimage.gaussian_filter(backward, sigma=1/0.29), 270))

    gt = np_gts.real
    ks = np_kss.real
    gan = np.mean(np_samps, axis=0).real

    r_ks.append(pearsoncoeff(gt, ks, mask))
    r_gan.append(pearsoncoeff(gt, gan, mask))

    rmse_ks.append(rmse(ks, gt, mask))
    rmse_gan.append(rmse(gan, gt, mask))

    psnr_ks.append(psnr(gt, ks, mask))
    psnr_gan.append(psnr(gt, gan, mask))
    abs_error_gan = np.abs(gan - gt)
    psnr_vals = []
    # pearson_vals = []
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

    true_mse = mse(np_gts, gan)
    collected_mse_values = np.array([mse(gan, sample) for sample in np_samps])
    for q in ecp_vals.keys():
        level = q/100
        quantile_mse = np.quantile(collected_mse_values, q=level)
        if true_mse < quantile_mse:
            mse_coverage[q] += 1

mse_quantiles = list(mse_coverage.keys())
mse_coverage_np = np.array([mse_coverage[q]/num_imgs for q in mse_quantiles])

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

with open("results_psnr.json", "w") as json_file:
    json.dump(results_dict, json_file)

# quantiles = list(ecp_vals.keys())
# ecp_means = [np.mean(ecp_vals[q]) for q in quantiles]

# results = {
#     "mse_coverage_np": mse_coverage_np,
#     "ecp_vals": ecp_vals,
#     "ecp_means": ecp_means,
#     "quantiles": quantiles,
# }

# np.save("/home/jjwhit/rcGAN/figures/coverage_results_1k.npy", results, allow_pickle=True)

# plt.figure(figsize=(8, 6))
# plt.plot(quantiles, ecp_means, marker='o', linestyle='-')
# plt.plot([0, 100], [0, 1], "k--", label="Ideal")  # Reference line for perfect calibration
# plt.xlabel("Credible Interval (%)")
# plt.ylabel("Empirical Coverage Probability")
# plt.title("ECP Plot")
# plt.legend()
# plt.grid()
# plt.savefig("/home/jjwhit/rcGAN/figures/ecp_plot_3.png")
# plt.close()

psnr_mean = np.mean(all_psnr_vals, axis=0)
psnr_std = np.std(all_psnr_vals, axis=0)
N_vals = np.arange(1, 33)
plt.figure(figsize=(8, 6))
plt.plot(N_vals, psnr_mean, color="black", label="Mean PSNR")
plt.fill_between(N_vals, psnr_mean - psnr_std, psnr_mean + psnr_std, color="blue", alpha=0.3, label="±1 Std Dev")
plt.xlabel("Number of Samples N")
plt.ylabel("PSNR")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("/home/jjwhit/rcGAN/figures/psnr_vs_samples.png")
plt.close()

# Plot: Number of samples vs Pearson Correlation
# pearson_mean = np.mean(all_pearson_vals, axis=0)
# pearson_std = np.std(all_pearson_vals, axis=0)
# plt.figure(figsize=(8, 6))
# plt.plot(N_vals, pearson_mean, color="black", label="Mean Pearson Corr.")
# plt.fill_between(N_vals, pearson_mean - pearson_std, pearson_mean + pearson_std, color="red", alpha=0.3, label="±1 Std Dev")
# plt.xlabel("Number of Samples N")
# plt.ylabel("Pearson Coefficient")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("/home/jjwhit/rcGAN/figures/pearson_vs_samples.png")
# plt.close()

# plt.figure(figsize=(8, 6))
# plt.plot(quantiles, mse_coverage_np, marker='o', linestyle='-')
# plt.plot([0, 100], [0, 1], "k--", label="Ideal")  # Reference line for perfect calibration
# plt.xlabel("Credible Interval (%)")
# plt.ylabel("Empirical Coverage Probability (l2 ball around posterior mean)")
# plt.title("ECP Plot")
# plt.legend()
# plt.grid()
# plt.savefig("/home/jjwhit/rcGAN/figures/ecp_mse_plot_1k.png")

