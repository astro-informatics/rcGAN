import torch
import yaml
import types
import json
import numpy as np
import os
import sys
sys.path.append("/home/jjwhit/rcGAN/")
from data.lightning.MassMappingDataModule import MMDataModule
from data.lightning.MassMappingDataModule import MMDataTransform 
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.mmGAN import mmGAN
from utils.mri import transforms
from scipy import ndimage


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
    fig_count = 1
    dm.setup()
    test_loader = dm.test_dataloader()

    with torch.no_grad():
        mmGAN_model = mmGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + "/checkpoint_best.ckpt"
        )

        mmGAN_model.cuda()

        mmGAN_model.eval()

        cosmos_map = np.load(cfg.cosmo_dir_path + 'cosmos_shear_cropped.npy', allow_pickle=True)
        ks = MMDataTransform.backward_model(cosmos_map, MMDataTransform.compute_fourier_kernel(300))
        pt_ks = transforms.to_tensor(ks)
        pt_ks = pt_ks.permute(2,0,1)
        cosmos_map = transforms.to_tensor(cosmos_map)
        cosmos_map = cosmos_map.permute(2,0,1)
        normalized_gamma, mean, std = transforms.normalise_complex(cosmos_map)
        normalized_ks, mean_ks, std_ks = transforms.normalise_complex(pt_ks)
        normalized_ks = transforms.normalize(pt_ks, mean_ks, std_ks)
        normalized_gamma = torch.cat([normalized_gamma, normalized_ks], dim=0).unsqueeze(0).float()
        batch_size = 9
        normalized_gamma = normalized_gamma.repeat(batch_size, 1, 1, 1)
        gens_mmGAN = torch.zeros(
            size=(batch_size, cfg.num_z_test, cfg.im_size, cfg.im_size)
        ).cuda()

        for z in range(cfg.num_z_test):
            gens_mmGAN[:, z, :, :] = mmGAN_model.reformat(mmGAN_model.forward(normalized_gamma.cuda())).squeeze(dim=-1)

        avg_mmGAN = torch.mean(gens_mmGAN, dim=1)
        for j in range(batch_size):
            np_avgs = {
                "mmGAN": None,
            }

            np_samps = {
                "mmGAN": [],
            }

            np_stds = {
                "mmGAN": None,
            }
            kappa_mean = cfg.kappa_mean
            kappa_std = cfg.kappa_std
            np_avgs["mmGAN"] = ndimage.rotate(
                (avg_mmGAN[j] * kappa_std + kappa_mean).cpu().numpy(), 180
            )
            for z in range(cfg.num_z_test):
                np_samps["mmGAN"].append(
                    ndimage.rotate(
                        (gens_mmGAN[j, z] * kappa_std + kappa_mean).cpu().numpy(),
                        180,
                    )
                )

            np_stds["mmGAN"] = np.std(np.stack(np_samps["mmGAN"]), axis=0)
        # check cfg.save_path + f"cosmos/ exists:

        if not os.path.exists(cfg.save_path + "cosmos/"):
            os.makedirs(cfg.save_path + "cosmos/")
            print("New COSMOS save directory created.")

        np.save(
            cfg.save_path + f"cosmos/np_avgs_cos.npy",
            np_avgs["mmGAN"],
        )
        np.save(
            cfg.save_path + f"cosmos/np_stds_cos.npy",
            np_stds["mmGAN"],
        )
        np.save(
            cfg.save_path + f"cosmos/np_samps_cos.npy",
            np_samps["mmGAN"],
        )

        if fig_count == args.num_figs:
            sys.exit()
        fig_count += 1
