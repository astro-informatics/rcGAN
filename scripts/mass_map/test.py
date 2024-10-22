import torch
import yaml
import types
import json
import lpips

import numpy as np

import sys

sys.path.append("/home/jjwhit/rcGAN/")

from data.lightning.MassMappingDataModule import MMDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.mmGAN import mmGAN
from utils.mri.math import tensor_to_complex_np
from mass_map_utils.scripts.ks_utils import psnr
from skimage.metrics import structural_similarity as ssim
from utils.embeddings import VGG16Embedding
from evaluation_scripts.mass_map_cfid.cfid_metric import CFIDMetric
from DISTS_pytorch import DISTS


def load_object(dct):
    return types.SimpleNamespace(**dct)


def rgb(im, im_size, unit_norm=False):
    """
    Args:
        im: Input image.
        im_size (int): Width of (square) image.
    """
    embed_ims = torch.zeros(size=(3, im_size, im_size))
    tens_im = torch.tensor(im)

    if unit_norm:
        tens_im = (tens_im - torch.min(tens_im)) / (
            torch.max(tens_im) - torch.min(tens_im)
        )
    else:
        tens_im = (
            2
            * (tens_im - torch.min(tens_im))
            / (torch.max(tens_im) - torch.min(tens_im))
            - 1
        )

    embed_ims[0, :, :] = tens_im
    embed_ims[1, :, :] = tens_im
    embed_ims[2, :, :] = tens_im

    return embed_ims.unsqueeze(0)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    config_path = args.config

    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    cfg.batch_size = cfg.batch_size * 4
    dm = MMDataModule(cfg)
    mask = np.load(
        "/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_mask.npy", allow_pickle=True
    ).astype(bool)

    dm.setup()

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    lpips_met = lpips.LPIPS(net="alex")
    dists_met = DISTS()

    with torch.no_grad():
        model = mmGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir
            + args.exp_name
            + "/checkpoint-epoch=93.ckpt"
        )
        model.cuda()
        model.eval()

        n_samps = [1, 2, 4, 8, 16, 32]

        for n in n_samps:
            print(f"\n\n{n} SAMPLES")
            psnrs = []
            ssims = []
            apsds = []
            lpipss = []
            distss = []

            for i, data in enumerate(test_loader):
                y, x, mean, std = data
                y = y.cuda()
                x = x.cuda()
                mean = mean.cuda()
                std = std.cuda()

                gens = torch.zeros(
                    size=(y.size(0), n, cfg.im_size, cfg.im_size)
                ).cuda()
                for z in range(n):
                    gens[:, z, :, :] = model.reformat(model.forward(y)).squeeze(-1)

                avg = torch.mean(gens, dim=1)

                gt = model.reformat(x).squeeze(-1)

                for j in range(y.size(0)):
                    single_samps = np.zeros((n, cfg.im_size, cfg.im_size))

                    kappa_mean = cfg.kappa_mean
                    kappa_std = cfg.kappa_std

                    gt_ksp, avg_ksp = (gt[j] * kappa_std + kappa_mean).cpu().numpy(), (
                        avg[j] * kappa_std + kappa_mean
                    ).cpu().numpy()

                    avg_gen_np = avg_ksp  # should be real
                    gt_np = gt_ksp  # should also be real already

    
                    for z in range(n):
                        np_samp = (
                            (gens[j, z, :, :] * kappa_std + kappa_mean).cpu().numpy()
                        )
                        np_samp = np_samp.squeeze()
                        single_samps[z, :, :] = np_samp

                    med_np = np.median(single_samps, axis=0)

                    apsds.append(np.mean(np.std(single_samps, axis=0), axis=(0, 1)))
                    psnrs.append(psnr(gt_np, avg_gen_np, mask))
                    ssims.append(ssim(gt_np, avg_gen_np))
                    lpipss.append(
                        lpips_met(
                            rgb(gt_np, cfg.im_size), rgb(avg_gen_np, cfg.im_size)
                        ).numpy()
                    )
                    distss.append(
                        dists_met(
                            rgb(gt_np, cfg.im_size, unit_norm=True),
                            rgb(avg_gen_np, cfg.im_size, unit_norm=True),
                        ).numpy()
                    )

            print("AVG Recon")
            print(
                f"PSNR: {np.mean(psnrs):.2f} \pm {np.std(psnrs) / np.sqrt(len(psnrs)):.2f}"
            )
            print(
                f"SSIM: {np.mean(ssims):.4f} \pm {np.std(ssims) / np.sqrt(len(ssims)):.4f}"
            )
            print(
                f"LPIPS: {np.mean(lpipss):.4f} \pm {np.std(lpipss) / np.sqrt(len(lpipss)):.4f}"
            )
            print(
                f"DISTS: {np.mean(distss):.4f} \pm {np.std(distss) / np.sqrt(len(distss)):.4f}"
            )
            print(f"APSD: {np.mean(apsds):.1f}")

    cfids = []
    m_comps = []
    c_comps = []

    inception_embedding = VGG16Embedding(parallel=True)
    # CFID_1
    cfid_metric = CFIDMetric(
        gan=model,
        loader=test_loader,
        image_embedding=inception_embedding,
        condition_embedding=inception_embedding,
        cuda=True,
        args=cfg,
        ref_loader=False,
        num_samps=32,
    )

    cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    cfids.append(cfid)
    m_comps.append(m_comp)
    c_comps.append(c_comp)

    # CFID_2
    cfid_metric = CFIDMetric(
        gan=model,
        loader=val_dataloader,
        image_embedding=inception_embedding,
        condition_embedding=inception_embedding,
        cuda=True,
        args=cfg,
        ref_loader=False,
        num_samps=8,
    )

    cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    cfids.append(cfid)
    m_comps.append(m_comp)
    c_comps.append(c_comp)

    # CFID_3
    cfid_metric = CFIDMetric(
        gan=model,
        loader=val_dataloader,
        image_embedding=inception_embedding,
        condition_embedding=inception_embedding,
        cuda=True,
        args=cfg,
        ref_loader=train_dataloader,
        num_samps=1,
    )

    cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    cfids.append(cfid)
    m_comps.append(m_comp)
    c_comps.append(c_comp)

    print("\n\n")
    for l in range(3):
        print(
            f"CFID_{l+1}: {cfids[l]:.2f}; M_COMP: {m_comps[l]:.4f}; C_COMP: {c_comps[l]:.4f}"
        )
