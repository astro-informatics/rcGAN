import argparse
import torch
import numpy as np
import yaml
import json
import types

import sys
dir = '/home/jjwhit/rcGAN/'
sys.path.append(dir)
from tqdm import tqdm 
from pytorch_lightning import seed_everything
from data.lightning.MassMappingDataModule import MMDataModule
from models.lightning.mmGAN import mmGAN
import os
import matplotlib.pyplot as plt


def load_object(dct):
    return types.SimpleNamespace(**dct)

def Li(lbda, x, xhat, xvar):
    return 1 - torch.sum(torch.where(torch.abs(x - xhat) <= (lbda * xvar), 1, 0)) / (
        x.shape[0] * x.shape[1]
    )

def ecp(samples, gt, lower_quant, upper_quant, mask):
    samples = samples[:,mask==1]
    gt = gt[mask==1]
    lower = np.percentile(samples, lower_quant)
    upper = np.percentile(samples, upper_quant)
    inside_interval = (gt >= lower) & (gt <= upper)
    return np.mean(inside_interval)


def RCPS(pred_list, N=500, risk=0.1, error=0.1, dl=0.01, lmax=3, verbose=False):
    """
    Risk-Controlling Prediction Set (RCPS)

    ucb: upper confidence bound on the risk
    dl: lambda step size
    pred_list:
        x: ground truth
        xhat: reconstruction
        xvar: Notion of UQ related to the Li() function

    Note
    ====
    See Algorithm 2 from arXiv:2202.05265 (Image-to-Image Regression with
    Distribution-Free Uncertainty Quantiﬁcation and Applications in Imaging)

    """
    lbda = lmax
    ucb = 0
    while ucb <= risk and lbda > dl:
        ucb = 0
        lbda = lbda - dl
        for i in range(N):
            x, xhat, xvar = pred_list[i]
            L = Li(lbda, x, xhat, xvar)
            ucb += L
        ucb /= N
        ucb += np.sqrt((1 / (2 * N)) * np.log(1 / error))
        if verbose:
            print(ucb)

    print("Calibration lambda: ", lbda + dl)
    return lbda + dl


def coverage_plots(args, device):

    # Define colors
    # cmap = mpl.colormaps.get_cmap("Set3")  # .resampled(6)
    # colors = cmap(np.arange(0, cmap.N))

    # Config stuff
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)
    
    # Randseed
    seed_everything(cfg.randseed, workers=True)

    dm = MMDataModule(cfg)
    dm.setup()

    test_loader = dm.test_dataloader()
    val_loader = dm.val_dataloader()

    # Load model
    model = mmGAN.load_from_checkpoint(
        checkpoint_path = cfg.checkpoint_dir + args.exp_name + "/checkpoint-epoch=93.ckpt"
    ).to(device).eval()

    percentiles = np.linspace(0.01, 0.99, cfg.n_percentiles)
    # lbda_list = []

    # print("Calculating conformalisation vals..")

    # for percent in percentiles:
    #     val_list = []

    #     for y, x, mean, std in tqdm(val_loader, desc=f"Processing percentile {percent}"):
    #         y, x = y.to(device), x.to(device)

    #         gens_mmGAN = torch.zeros(
    #             size=(y.size(0), cfg.num_z_test, cfg.im_size, cfg.im_size)
    #         ).cuda()

    #         with torch.no_grad():
    #             for z in range(cfg.num_z_test):
    #                 gens_mmGAN[:, z, :, :] = model.reformat(model.forward(y)).squeeze(-1)
    #             xhat = torch.mean(gens_mmGAN, dim=1)
    #             xvar = torch.std(gens_mmGAN, dim=1)
            
    #         for i in range(x.shape[0]):
    #             # TODO: Check these are the same size
    #             val_list.append((x[i], xhat[i], xvar[i].squeeze()))


    #     lbda = RCPS(val_list, error=0.1, risk=1 - percent, N=len(val_list), lmax=3)
    #     lbda_list.append(lbda)
    #     print(f"q: {percent}, lambda: {lbda}")

    # # Save conformalisation values
    # np.save(
    #     cfg.save_path + "coverage_results.npy", {
    #         "config": cfg,
    #         "lbda_list": np.array(lbda_list),
    #         "percentiles": percentiles,
    #     }
    # )
    # print("Conformalisation calculations done :)")

    # compare original coverage with new coverage
    print("Making new coverage plots...")
    lambda_data = np.load(cfg.save_path + "coverage_results.npy", allow_pickle=True).item()
    lbda_list = lambda_data["lbda_list"]
    ecp_orig = [[] for _ in percentiles] 
    ecp_rcps = [[] for _ in percentiles] 
    mask = np.load("/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_mask.npy", allow_pickle=True).astype(bool)
    for y, x, mean, std in tqdm(test_loader):
        y, x = y.to(device), x.to(device)
        # (batch, samples, 300, 300)
        gens_mmGAN = torch.zeros(
            size=(y.size(0), cfg.num_z_test, cfg.im_size, cfg.im_size)
        ).cuda()

        with torch.no_grad():
            for z in range(cfg.num_z_test):
                gens_mmGAN[:, z, :, :] = model.reformat(model.forward(y)).squeeze(-1)
        x_np = x.squeeze(1).cpu().numpy()
        # avg_mmGAN = torch.mean(gens_mmGAN, dim=1)
        # avg_mmGAN = avg_mmGAN.cpu().numpy()  
        samps = gens_mmGAN.cpu().numpy()  # shape: (9, 32, 300, 300)
        mean_pred = np.mean(samps, axis=1)  # shape: (9, 300, 300)
        std_pred = np.std(samps, axis=1)    # shape: (9, 300, 300)
        # calculating for each percentile (10 of them)
        for p_idx, percent in enumerate(percentiles):
            lower_q = (1 - percent) / 2 * 100
            upper_q = (1 + percent) / 2 * 100
            lbda = lbda_list[p_idx]
            for i in range(y.size(0)):
                # looping over the batch
                orig_ecp_val = ecp(samps[i], x_np[i], lower_q, upper_q, mask) # [0] FOR TESTING ONLY!
                ecp_orig[p_idx].append(orig_ecp_val)
                lower = mean_pred[i] - (std_pred[i] + lbda)
                upper = mean_pred[i] + (std_pred[i] + lbda)
                inside = (x_np[i] >= lower) & (x_np[i] <= upper)
                rcps_val = np.mean(inside[mask])
                ecp_rcps[p_idx].append(rcps_val)
    mean_orig = [np.mean(vals) for vals in ecp_orig]
    mean_rcps = [np.mean(vals) for vals in ecp_rcps]

    plt.figure(figsize=(8, 6))
    plt.plot(percentiles, mean_orig, label='Original ECP', linestyle='--', color='blue')
    plt.plot(percentiles, mean_rcps, label='RCPS Adjusted ECP', color='red')
    plt.plot(percentiles, percentiles, label='Ideal Coverage', linestyle=':', color='black')
    plt.xlabel("Target Coverage Level")
    plt.ylabel("Empirical Coverage")
    plt.title("Empirical Coverage Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(cfg.save_path + "ecp_comparison_plot.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantile comparison")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file")
    parser.add_argument("-e", "--exp-name", type=str, required=True, help="Experiment name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coverage_plots(args, device)