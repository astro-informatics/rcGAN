import numpy as np
import json
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/jjwhit/rcGAN")
from data.lightning.MassMappingDataModule import MMDataTransform


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

lambdas = np.linspace(0.7, 3.0, 40)
levels = np.arange(0.05, 1.0, 0.05)
results = {}

def hoeffding_naive_ucb(losses, n, delta, maxiters=None):
    R_hat = np.mean(losses)
    return R_hat + np.sqrt((1 / (2 * n)) * np.log(1 / delta))

def coverage_indicators(samples, gt, mask, level, lam=1.0):
    """
    Returns Bernoulli coverage indicators (1 = covered, 0 = not covered)
    for all unmasked pixels.
    """
    lower_q = 100 * (1 - level) / 2
    upper_q = 100 - lower_q

    lower = np.percentile(samples, lower_q, axis=0)
    upper = np.percentile(samples, upper_q, axis=0)

    mean = np.mean(samples, axis=0)
    lower_scaled = mean + lam * (lower - mean)
    upper_scaled = mean + lam * (upper - mean)

    covered = (gt >= lower_scaled) & (gt <= upper_scaled)
    covered = covered[mask == 1]

    return covered.astype(float)

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

    #Scaling by lambda calculated during RCPS calibration
    mean = np.mean(samples, axis=0)
    lower_scaled = mean + lam*(lower - mean)
    upper_scaled = mean + lam*(upper - mean)

    inside_interval = (gt >= lower_scaled) & (gt <= upper_scaled)
    inside_interval = inside_interval[mask==1]
    return np.mean(inside_interval)

def bisection_lambda_ecp_hoeffding(
    coverage_func,
    samples,
    gt,
    mask,
    level,
    start_interval,
    delta=0.05,
    tol=1e-4,
    verbose=True,
):
    lam_low, lam_high = start_interval

    def compute_coverage_lcb(lam):
        all_losses = []

        for i in range(len(gt)):
            cov = coverage_func(samples[i], gt[i], mask, level, lam)
            losses = 1.0 - cov  # miscoverage
            all_losses.append(losses)

        all_losses = np.concatenate(all_losses)
        n = len(all_losses)

        R_ucb = hoeffding_naive_ucb(
            losses=all_losses,
            n=n,
            delta=delta,
            maxiters=None,
        )

        return 1.0 - R_ucb  # coverage lower bound

    lcb_low = compute_coverage_lcb(lam_low) - level
    lcb_high = compute_coverage_lcb(lam_high) - level

    if np.sign(lcb_low) == np.sign(lcb_high):
        raise ValueError("Bisection requires λ to bracket target coverage")

    iter_count = 0

    while True:
        lam_mid = 0.5 * (lam_low + lam_high)
        lcb_mid = compute_coverage_lcb(lam_mid)
        diff = lcb_mid - level
        iter_count += 1

        if verbose:
            print(
                f"[Iter {iter_count}] λ={lam_mid:.8f}, "
                f"coverage LCB={lcb_mid:.5f}, diff={diff:.5e}"
            )

        if abs(diff) < tol:
            if verbose:
                print(f"Converged at λ={lam_mid:.8f}")
            break

        if np.sign(diff) == np.sign(lcb_low):
            lam_low = lam_mid
            lcb_low = diff
        else:
            lam_high = lam_mid
            lcb_high = diff

    return lam_mid, lcb_mid



calibrated_lambdas = {}
calibrated_ecp = {}

all_samps = []
all_gts = []
for map in range(1, 101):
    np_gts = np.load(data_dir + f"kappa/np_gt_{map:04d}.npy")
    np_samps = np.load(data_dir + f"recon/np_samps_{map:04d}.npy")
    gt = np_gts.real
    samples = np_samps.real
    all_samps.append(np_samps)
    all_gts.append(np_gts)
all_gts = np.stack(all_gts)
all_samps = np.stack(all_samps)

for level in levels:
    print(f"Starting at level {level}:")
    prev_val = None
    prev_lam = None
    bracket_found = False
    results[level] = {}

    def compute_coverage_lcb(lam):
        all_losses = []
        for i in range(len(all_gts)):
            cov = coverage_indicators(all_samps[i], all_gts[i], mask, level, lam)
            losses = 1.0 - cov
            all_losses.append(losses)
        all_losses = np.concatenate(all_losses)
        n = len(all_losses)
        R_ucb = hoeffding_naive_ucb(all_losses, n, delta=0.05)
        return 1.0 - R_ucb

    for lam in lambdas:
        lcb = compute_coverage_lcb(lam)
        results[level][str(round(lam, 8))] = {
            "level": level,
            "lambda": lam,
            "coverage_lcb": float(lcb),
        }

        print(f"lambda={lam:.8f} => Coverage LCB: {lcb:.6f}")

        if prev_val is not None and not bracket_found:
            if (prev_val < level and lcb > level) or (prev_val > level and lcb < level):
                lam_low = prev_lam
                lam_high = lam
                bracket_found = True
                print(f"Bracket found between λ={lam_low:.6f} and λ={lam_high:.6f}")
                break

        prev_val = lcb
        prev_lam = lam

    if not bracket_found:
        raise RuntimeError(
            "No bracketing λ pair found for Hoeffding LCB. Increase lambda range."
        )


    # Run bisection algorithm
    # best_lam, best_ecp = bisection_lambda_ecp(ecp, all_samps, all_gts, mask, level, start_interval=(lam_low, lam_high), tol=0.001, verbose=True)
    best_lam, best_lcb = bisection_lambda_ecp_hoeffding(coverage_indicators, all_samps, all_gts, mask, level, start_interval=(lam_low, lam_high), delta=0.05)
    print(f"\n✅ Calibrated lambda: λ ≈ {best_lam:.8f}")
    calibrated_lambdas[level] = best_lam
    # calibrated_ecp[level] = best_ecp
    calibrated_ecp[level] = best_lcb

    # Plot lambda vs empirical coverage
    # lambda_vals = [results[level][key]["lambda"] for key in results[level]]
    # ecp_vals = [results[level][key]["empirical_coverage"] for key in results[level]]

    # plt.figure(figsize=(8, 5))
    # plt.plot(lambda_vals, ecp_vals, marker='o', linestyle='-')
    # plt.axhline(y=0.9, color='r', linestyle='--', label=f'Target Coverage = {level.2f}')
    # plt.xlabel("Lambda")
    # plt.ylabel("Empirical Coverage Probability")
    # plt.title(f"Lambda vs Empirical Coverage ({level.2f} CI)")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"/home/jjwhit/rcGAN/figures/lambda_vs_ecp_lvl_{int(level*100)}.png")
    # plt.show()

ecp_uncalibrated = {}
for level in levels:
    ecp_vals_uc = []
    for i in range(len(all_gts)):
        ecp_val = ecp(all_samps[i], all_gts[i], mask, level)
        ecp_vals_uc.append(ecp_val)
    mean_ecp = np.mean(ecp_vals_uc)
    ecp_uncalibrated[level] = mean_ecp
    print(f"Level {level:.2f}: Uncalibrated ECP = {mean_ecp:.4f}")

# final cov. plot
levels = [l for l in levels]
ecp_uncal = [ecp_uncalibrated[l] for l in levels]
ecp_cal = [calibrated_ecp[l] for l in levels]

plt.figure(figsize=(8, 6))
plt.plot(levels, ecp_uncal, marker='o', linestyle='-', label='Uncalibrated')
plt.plot(levels, ecp_cal, marker='s', linestyle='-', label='Calibrated')
plt.plot([0, 1], [0, 1], "k--", label="Ideal")

plt.xlabel("Credible Interval (%)")
plt.ylabel("Empirical Coverage Probability")
plt.title("ECP Curves (Calibrated vs Uncalibrated)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("/home/jjwhit/rcGAN/figures/ecp_hoeff.png")
plt.close()

final_results = []
for level in levels:
    final_results.append({
        "level": float(level),
        "lambda": float(calibrated_lambdas[level]),
        "ecp": float(calibrated_ecp[level]),
    })

# Save the list to JSON
with open("/home/jjwhit/rcGAN/hoeffding.json", "w") as f:
    json.dump(final_results, f, indent=4)