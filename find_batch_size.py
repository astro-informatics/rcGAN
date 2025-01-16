"""This script can be used to find the maximum batch size that fits on a single GPU. Make sure to run this on the same GPU as you run the full test run.
"""

import torch
import torch.nn as nn
import yaml
import types
import json

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from data.lightning.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from models.lightning.rcGAN import rcGAN
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from data.lightning.MassMappingDataModule import MMDataModule
from data.lightning.RadioDataModule import RadioDataModule
from models.lightning.mmGAN import mmGAN
from torch.utils.data import DataLoader

def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)

    print(f"Experiment Name: {args.exp_name}")
    print(f"Number of GPUs: {args.num_gpus}")
    print("Device count: ",torch.cuda.device_count())
    print(f"Config file path: {args.config}")

    config_path = args.config

    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        # Load the correct model
        if cfg.experience == 'mri':
            DM = MRIDataModule
            model = rcGAN(cfg, args.exp_name, args.num_gpus)
        elif cfg.experience == 'mass_mapping':
            DM = MMDataModule
            model = mmGAN(cfg, args.exp_name, args.num_gpus)
        elif cfg.experience == 'radio':
            DM = RadioDataModule
            model = mmGAN(cfg, args.exp_name, args.num_gpus)
        else:
            print("No valid experience selected in config file. Options are 'mri', 'mass_mapping', 'radio'.")
            exit()


    class newDataModule(DM):
        def __init__(self, cfg, batch_size):
            super().__init__(cfg)
            self.args.batch_size = batch_size
            self.hparams.batch_size = batch_size
    
        def train_dataloader(self):
            return DataLoader(
                dataset=self.train,
                batch_size=self.hparams.batch_size,
                num_workers=self.args.num_workers,
                drop_last=True,
                pin_memory=False
           )
        
    test_model = mmGAN(cfg, args.exp_name, num_gpus=1)
    dm = newDataModule(cfg, cfg.batch_size)
    trainer = pl.Trainer(accelerator="gpu", devices=1,auto_scale_batch_size="binsearch")
    trainer.tune(test_model, datamodule=dm)

    print("="*20)
    print("Maximum Batch Size: ", dm.hparams.batch_size)
    print("="*20)