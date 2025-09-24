import torch
import yaml
import types
import json

import numpy as np
import matplotlib.patches as patches

import sys
sys.path.append('/home/mars/git/rcGAN/')

from data.lightning.RadioDataModule import RadioDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.riGAN import riGAN
from models.lightning.GriGAN import GriGAN

from utils.mri.math import tensor_to_complex_np
import sys
from datetime import date
import pickle
import os


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

    dm = RadioDataModule(cfg)
    fig_count = 5
    dm.setup()
    test_loader = dm.test_dataloader()

    today = date.today()
    pred_dir = cfg.data_path + "/pred/"
    os.makedirs(pred_dir, exist_ok=True)
    
    
    with torch.no_grad():
        if cfg.__dict__.get("gradient", False):
            RIGAN_model = GriGAN.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')
        else:
            RIGAN_model = riGAN.load_from_checkpoint(
                checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')
        
        RIGAN_model.cuda()

        RIGAN_model.eval()

        true = []
        dirty = []
        pred = []
       	
#         cfg.num_z_test = 100
        
        for i, data in enumerate([0]):
            print(f"{i}/{len([0])}")
            im, d, p, _, _ = np.load('/home/mars/src_aiai/notebooks/GAN_30Dor.npy')
            
            pt_y, pt_x, mean, std = dm.test.transform((im, d, p))
            
            y = torch.tensor(pt_y[None,:]).cuda()
            x = torch.tensor(pt_x[None, :]).cuda()
            mean = torch.tensor(np.array([mean])).cuda()
            std = torch.tensor(np.array([std])).cuda()  
            
            gens_RIGAN = torch.zeros(size=(y.size(0), cfg.num_z_test, cfg.im_size, cfg.im_size, 1)).cuda()

            for z in range(cfg.num_z_test):
                gens_RIGAN[:, z, :, :, :] = RIGAN_model.reformat(RIGAN_model.forward(y))

            
            avg_RIGAN = torch.mean(gens_RIGAN, dim=1)

            gt = RIGAN_model.reformat(x)
            zfr = RIGAN_model.reformat(y)
            

            tensor_to_complex_np = lambda x: x
            for j in range(y.size(0)):
            
                true.append( torch.tensor(tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu())).numpy().real)
                dirty.append( torch.tensor(tensor_to_complex_np((zfr[j] * std[j] + mean[j]).cpu())).numpy().real )
                pred.append(torch.tensor(tensor_to_complex_np((gens_RIGAN[j] * std[j] + mean[j]).cpu())).numpy().real )
                
                pickle.dump([np.array(true), np.array(dirty), np.array(pred)], open(f"{pred_dir}/pred_30DOR_{args.exp_name}_{today}.pkl", "wb"))
                
                exit()
         

        

        

       	
