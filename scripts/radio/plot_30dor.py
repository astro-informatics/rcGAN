import torch
import yaml
import types
import json

import numpy as np
import matplotlib.patches as patches

import sys
sys.path.append('/home/mars/git/rcGAN/')

from data.lightning.MassMappingDataModule import MMDataModule
from data.lightning.RadioDataModule import RadioDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.riGAN import riGAN
from models.lightning.GriGAN import GriGAN

from utils.mri.math import tensor_to_complex_np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import ndimage
import sys
from datetime import date
import pickle
import os

from utils.mri import transforms


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    config_path = args.config

    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

#     dm = MMDataModule(cfg)
    dm = RadioDataModule(cfg)
    fig_count = 5
    dm.setup()
#     train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    today = date.today()
    pred_dir = cfg.data_path + "/pred/"
    os.makedirs(pred_dir, exist_ok=True)
    
    
    with torch.no_grad():
#         mmGAN_model = mmGAN.load_from_checkpoint(
#             checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')

        if cfg.__dict__.get("gradient", False):
            mmGAN_model = GriGAN.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')
        else:
            mmGAN_model = riGAN.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')
        
        mmGAN_model.cuda()

        mmGAN_model.eval()

        true = []
        dirty = []
        pred = []
       	
#         cfg.num_z_test = 100
        
        for i, data in enumerate([0]):
            print(f"{i}/{len([0])}")
            im, d, p, _, _ = np.load('/home/mars/src_aiai/notebooks/GAN_30Dor.npy')
            
            pt_y, pt_x, mean, std = dm.test.transform((im, d, p))
#             mean, std = np.mean(d), np.std(d)
#             x = im # (im - mean)/std
#             y = d #(d - mean)/std
#             uv = p #(p - np.mean(p))/np.std(p)
#             mean = 0
#             std = 1

            
            
#             # Format input gt data.
#             pt_x = transforms.to_tensor(x)[:, :, None] # Shape (H, W, 2)
#             pt_x = pt_x.permute(2, 0, 1)  # Shape (2, H, W)
#             # Format observation data.
#             pt_y = transforms.to_tensor(y)[:, :, None] # Shape (H, W, 2)
#             pt_y = pt_y.permute(2, 0, 1)  # Shape (2, H, W)
#             # Format uv data
#             pt_uv = transforms.to_tensor(uv)[:, :, None] # Shape (H, W, 1)
#             pt_uv = pt_uv.permute(2, 0, 1)  # Shape (1, H, W)
#             # Normalize everything based on measurements y

#             pt_x = pt_x.float()
#             pt_y = torch.cat([pt_y, pt_uv], dim=0).float()
            
            y = torch.tensor(pt_y[None,:]).cuda()
            x = torch.tensor(pt_x[None, :]).cuda()
            mean = torch.tensor(np.array([mean])).cuda()
            std = torch.tensor(np.array([std])).cuda()  
            
            gens_mmGAN = torch.zeros(size=(y.size(0), cfg.num_z_test, cfg.im_size, cfg.im_size, 1)).cuda()

            for z in range(cfg.num_z_test):
                gens_mmGAN[:, z, :, :, :] = mmGAN_model.reformat(mmGAN_model.forward(y))

            
            avg_mmGAN = torch.mean(gens_mmGAN, dim=1)

            gt = mmGAN_model.reformat(x)
            zfr = mmGAN_model.reformat(y)
            

            tensor_to_complex_np = lambda x: x
            for j in range(y.size(0)):
            
                true.append( torch.tensor(tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu())).numpy().real)
                dirty.append( torch.tensor(tensor_to_complex_np((zfr[j] * std[j] + mean[j]).cpu())).numpy().real )
                pred.append(torch.tensor(tensor_to_complex_np((gens_mmGAN[j] * std[j] + mean[j]).cpu())).numpy().real )
                
                pickle.dump([np.array(true), np.array(dirty), np.array(pred)], open(f"{pred_dir}/pred_30DOR_{args.exp_name}_{today}.pkl", "wb"))
                
                exit()
         

        

        

       	
