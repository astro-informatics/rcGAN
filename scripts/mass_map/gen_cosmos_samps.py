import torch
import yaml
import types
import json

import numpy as np

import sys
sys.path.append('/home/jjwhit/rcGAN/')

from data.lightning.MassMappingDataModule import MMDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.mmGAN import mmGAN
from utils.mri import transforms

def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":

    #TODO: Step 1: Load model.
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    config_path = args.config

    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    dm = MMDataModule(cfg)
    fig_count = 1
    dm.setup()
    test_loader = dm.test_dataloader()

    with torch.no_grad():
        mmGAN_model = mmGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')

        mmGAN_model.cuda()

        mmGAN_model.eval()

#Step 2: Load cosmos shear map
    cosmos_shear = np.load('/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_shear_cropped.npy')
    cosmos_shear_tensor = transforms.to_tensor(cosmos_shear)
    cosmos_shear_tensor = cosmos_shear_tensor.permute(2, 0, 1)
    cosmos_shear_tensor = cosmos_shear_tensor[None,:, :, :].cuda()

#Step 3: Feed through GAN and generate 32 samples.

    #TODO: Load the mean and std.
    normalized_gamma, mean, std = transforms.normalize_instance(cosmos_shear_tensor)
    normalized_gamma = normalized_gamma.cuda()
    mean = mean.cuda()
    std = std.cuda()

    gens_mmGAN = torch.zeros(size=(cfg.num_z_test, cfg.im_size, cfg.im_size, 2)).cuda()
    for z in range(cfg.num_z_test):
        gens_mmGAN[z, :, :, :] = mmGAN_model.reformat(mmGAN_model.forward(normalized_gamma).cuda())

    torch.save('/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_samps_no_norm', gens_mmGAN)
