import torch
import os
import yaml
import types
import json
import numpy as np
import shutil
import sys

sys.path.append(
    "/lustre/fswork/projects/rech/rbn/ulx23va/projects/unrolled_cGAN/repos/rcGAN/"
)

from tqdm import tqdm
from data.lightning.RadioDataModule import RadioDataModule
from data.lightning.MassMappingDataModule import MMDataModule
from utils.parse_args import create_arg_parser
from models.lightning.riGAN import riGAN
from models.lightning.GriGAN import GriGAN

from pytorch_lightning import seed_everything
from utils.embeddings import VGG16Embedding
from evaluation_scripts.radio_cfid.cfid_metric import CFIDMetric
from torchmetrics.functional import peak_signal_noise_ratio


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

    cfg.batch_size = 1

    if cfg.experience == "radio":
        dm = RadioDataModule(cfg)
    elif cfg.experience == "mass_mapping":
        dm = MMDataModule(cfg)
    else:
        exit("no data for specified experience")

    dm.setup()
    #     train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    best_cfid_epoch = -1
    best_psnr_epoch = -1
    inception_embedding = VGG16Embedding()
    best_cfid = 10000000
    best_psnr = -1
    start_epoch = 50  # Will start saving models after 50 epochs
    end_epoch = 100

    with torch.no_grad():

        for epoch in range(end_epoch):
            print(f"VALIDATING EPOCH: {epoch + 1}")
            try:
                if cfg.__dict__.get("gradient", False):
                    model = GriGAN.load_from_checkpoint(
                        checkpoint_path=cfg.checkpoint_dir
                        + args.exp_name
                        + f"/checkpoint-epoch={epoch}.ckpt"
                    )
                else:
                    model = riGAN.load_from_checkpoint(
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

            # Load the model
            model = model.cuda()
            model.eval()

            # Compute CFID
            cfid_metric = CFIDMetric(
                gan=model,
                loader=val_loader,
                image_embedding=inception_embedding,
                condition_embedding=inception_embedding,
                cuda=True,
                args=cfg,
                ref_loader=False,
                num_samps=1,
            )

            cfids = cfid_metric.get_cfid_torch_pinv()

            cfid_val = np.mean(cfids)

            if cfid_val < best_cfid:
                best_cfid_epoch = epoch
                best_cfid = cfid_val

            tensor_to_complex_np = lambda x: x
            psnr_values = []
            for i, data in tqdm(
                enumerate(val_loader), desc="Evaluating samples", total=len(val_loader)
            ):
                y, x, mean, std = data
                y = y.cuda()
                x = x.cuda()
                mean = mean.cuda()
                std = std.cuda()

                gens_RIGAN = torch.zeros(
                    size=(y.size(0), cfg.num_z_test, cfg.im_size, cfg.im_size, 1)
                ).cuda()

                for z in range(cfg.num_z_test):
                    gens_RIGAN[:, z, :, :, :] = model.reformat(model.forward(y))

                avg_RIGAN = torch.mean(gens_RIGAN, dim=1)

                gt = model.reformat(x)
                zfr = model.reformat(y)

                for j in range(y.size(0)):
                    psnr_values.append(peak_signal_noise_ratio(gt, avg_RIGAN).item())

            psnr_val = np.mean(psnr_values)
            if psnr_val > best_psnr:
                best_psnr_epoch = epoch
                best_psnr = psnr_val

    print(f"BEST CFID EPOCH: {best_cfid_epoch}")
    print(f"BEST PSNR EPOCH: {best_psnr_epoch}")
    # Best CFID epoch is saved
    shutil.copyfile(
        cfg.checkpoint_dir
        + args.exp_name
        + f"/checkpoint-epoch={best_cfid_epoch}.ckpt",
        cfg.checkpoint_dir + args.exp_name + f"/checkpoint_best_CFID.ckpt",
    )

    # Best PSNR epoch is saved
    shutil.copyfile(
        cfg.checkpoint_dir
        + args.exp_name
        + f"/checkpoint-epoch={best_psnr_epoch}.ckpt",
        cfg.checkpoint_dir + args.exp_name + f"/checkpoint_best_PSNR.ckpt",
    )
