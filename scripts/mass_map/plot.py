import torch
import yaml
import types
import json

import numpy as np

import sys

sys.path.append("/home/jjwhit/rcGAN/")

from data.lightning.MassMappingDataModule import MMDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.mmGAN import mmGAN
from utils.mri.math import tensor_to_complex_np
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

        for i, data in enumerate(test_loader):
            y, x, mean, std = data
            y = y.cuda()
            x = x.cuda()
            mean = mean.cuda()
            std = std.cuda()

            gens_mmGAN = torch.zeros(
                size=(y.size(0), cfg.num_z_test, cfg.im_size, cfg.im_size, 2)
            ).cuda()  # y.size(0) = batch size

            for z in range(cfg.num_z_test):
                gens_mmGAN[:, z, :, :, :] = mmGAN_model.reformat(mmGAN_model.forward(y))

            avg_mmGAN = torch.mean(gens_mmGAN, dim=1)

            gt = mmGAN_model.reformat(x)
            zfr = mmGAN_model.reformat(y)

            for j in range(y.size(0)):
                np_avgs = {
                    "mmGAN": None,
                }

                np_samps = {
                    "mmGAN": [],
                }

                np_stds = {
                    "mmGAN": None,
                }

                np_gt = None

                kappa_mean = cfg.kappa_mean
                kappa_std = cfg.kappa_std

                np_gt = ndimage.rotate(
                    (gt[j] * kappa_std + kappa_mean).squeeze().cpu().numpy(), 180
                )
                np_zfr = ndimage.rotate(
                    torch.tensor(
                        tensor_to_complex_np((zfr[j] * kappa_std + kappa_mean).cpu())
                    ).numpy(),
                    180,
                )

                np_avgs["mmGAN"] = ndimage.rotate(
                    (avg_mmGAN[j] * kappa_std + kappa_mean).squeeze().cpu().numpy(), 180
                )
                for z in range(cfg.num_z_test):
                    np_samps["mmGAN"].append(
                        ndimage.rotate(
                            (gens_mmGAN[j, z] * kappa_std + kappa_mean).squeeze().cpu().numpy(),
                            180,
                        )
                    )

                np_stds["mmGAN"] = np.std(np.stack(np_samps["mmGAN"]), axis=0)

                # Save arrays - gt, avg, samps, std, zfr
                np.save(f"/share/gpu0/jjwhit/samples/np_gt_{fig_count}.npy", np_gt)
                np.save(f"/share/gpu0/jjwhit/samples/np_zfr_{fig_count}.npy", np_zfr)
                np.save(
                    f"/share/gpu0/jjwhit/samples/np_avgs_{fig_count}.npy",
                    np_avgs["mmGAN"],
                )
                np.save(
                    f"/share/gpu0/jjwhit/samples/np_stds_{fig_count}.npy",
                    np_stds["mmGAN"],
                )
                np.save(
                    f"/share/gpu0/jjwhit/samples/np_samps_{fig_count}.npy",
                    np_samps["mmGAN"],
                )

                if fig_count == args.num_figs:
                    sys.exit()
                fig_count += 1
