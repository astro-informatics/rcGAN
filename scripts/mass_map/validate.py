import torch
import os
import yaml
import types
import json
import numpy as np
from tqdm import tqdm
from scipy import ndimage

import sys

sys.path.append("/home/jjwhit/rcGAN/")

from mass_map_utils.scripts.ks_utils import pearsoncoeff, psnr, snr, rmse
from data.lightning.MassMappingDataModule import MMDataModule
from utils.parse_args import create_arg_parser
from models.lightning.mmGAN import mmGAN
from pytorch_lightning import seed_everything


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    config_path = args.config

    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    dm = MMDataModule(cfg)
    dm.setup()
    val_loader = dm.val_dataloader()
    best_epoch = -1
    best_pearson = -1
    best_psnr = -1
    best_snr = -1
    best_rmse = 10000000
    start_epoch = 80  # Will start saving models after 80 epochs
    end_epoch = cfg.num_epochs
    mask = np.load(
        "/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_mask.npy", allow_pickle=True
    ).astype(bool)

    psnr_vals = []
    snr_vals = []
    rmse_vals = []
    pearson_vals = []

    with torch.no_grad():

        for epoch in range(start_epoch, end_epoch):
            print(f"VALIDATING EPOCH: {epoch}")

            # Loads the model one epoch at a time
            try:
                model = mmGAN.load_from_checkpoint(
                    checkpoint_path=cfg.checkpoint_dir
                    + args.exp_name
                    + f"/checkpoint-epoch={epoch}.ckpt"
                )
            except Exception as e:
                print(e)
                continue

            if model.is_good_model == 0:
                print("NO GOOD: SKIPPING...")
                continue

            model = model.cuda()
            model.eval()

            for i, data in tqdm(
                enumerate(val_loader), desc="Evaluating samples", total=len(val_loader)
            ):
                y, x, mean, std = data
                y = y.cuda()
                x = x.cuda()
                mean = mean.cuda()
                std = std.cuda()

                gens = torch.zeros(
                    size=(y.size(0), cfg.num_z_test, cfg.im_size, cfg.im_size)
                ).cuda() 
                for z in range(cfg.num_z_test):
                    gens[:, z, :, :] = model.reformat(model.forward(y)).squeeze(-1) # remove dimension that used to handle real/complex

                torch_reconstruction = torch.mean(gens, dim=1)
                torch_truth = model.reformat(x).squeeze(-1) # also removing extra dimension here too
                kappa_mean = cfg.kappa_mean
                kappa_std = cfg.kappa_std

                for j in range(y.size(0)):
                    reconstruction = ndimage.rotate(
                        (torch_reconstruction[j] * kappa_std + kappa_mean)
                        .squeeze().cpu()
                        .numpy(),
                        180,
                    )
                    truth = ndimage.rotate(
                        (torch_truth[j] * kappa_std + kappa_mean).squeeze().cpu().numpy(),
                        180,
                    )
                    reconstruction = np.real(
                        reconstruction
                    )  # recon and truth should already be real, but just in case
                    truth = np.real(truth)

                    pearson_val = pearsoncoeff(truth, reconstruction, mask)
                    pearson_vals.append((epoch, pearson_val))
                    if pearson_val > best_pearson:
                        best_epoch_pearson = epoch
                        best_pearson = pearson_val

                    psnr_val = psnr(truth, reconstruction, mask)
                    psnr_vals.append((epoch, psnr_val))
                    if psnr_val > best_psnr:
                        best_epoch_psnr = epoch
                        best_psnr = psnr_val

                    snr_val = snr(truth, reconstruction, mask)
                    snr_vals.append((epoch, snr_val))
                    if snr_val > best_snr:
                        best_epoch_snr = epoch
                        best_snr = snr_val

                    rmse_val = rmse(truth, reconstruction, mask)
                    rmse_vals.append((epoch, rmse_val))
                    if rmse_val < best_rmse:
                        best_epoch_rmse = epoch
                        best_rmse = rmse_val

    print(f"BEST EPOCH FOR PSNR: {best_epoch_psnr}")
    print(f"BEST EPOCH FOR SNR: {best_epoch_snr}")
    print(f"BEST EPOCH FOR RMSE: {best_epoch_rmse}")
    print(f"BEST EPOCH FOR R: {best_epoch_pearson}")

    print("Epoch | PSNR | SNR | RMSE | R")
    for epoch, psnr, snr, rmse, r in zip(
        range(start_epoch, end_epoch),
        psnr_vals,
        snr_vals,
        rmse_vals,
        pearson_vals,
    ):
        print(f"{epoch} | {psnr} | {snr} | {rmse} | {r}")

    # for epoch in range(end_epoch):
    #     try:
    #         if epoch != best_epoch:
    #             os.remove(cfg.checkpoint_dir + args.exp_name + f'/checkpoint-epoch={epoch}.ckpt')
    #     except:
    #         pass

    # os.rename(
    #     cfg.checkpoint_dir + args.exp_name + f"/checkpoint-epoch={best_epoch}.ckpt",
    #     cfg.checkpoint_dir + args.exp_name + f"/checkpoint_best.ckpt",
    # )
