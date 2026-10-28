import torch
import yaml
import types
import json
import sys
import os
sys.path.append('/home/jjwhit/rcGAN/')

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
# from data.lightning.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from models.lightning.rcGAN import rcGAN
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from data.lightning.MassMappingDataModule import MMDataModule
from data.lightning.RadioDataModule import RadioDataModule
from models.lightning.mmGAN import mmGAN
from models.lightning.riGAN import riGAN
from models.lightning.GriGAN import GriGAN


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(args.seed, workers=True)

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
            dm = MRIDataModule(cfg)
            model = rcGAN(cfg, args.exp_name, args.num_gpus)
        elif cfg.experience == 'mass_mapping':
            dm = MMDataModule(cfg)
            ms_level_weights = None
            if args.ms_sigma_file is not None:
                with open(args.ms_sigma_file) as f:
                    ms_level_weights = json.load(f)["weights"]
            model = mmGAN(cfg, args.exp_name, args.num_gpus,
                           crps_mode=args.crps_mode, crps_alpha=args.crps_alpha,
                           adv_scale=args.adv_scale, ms_level_weights=ms_level_weights)
        elif cfg.experience == 'radio':
            cfg.num_workers = args.num_gpus # set number of workers to same as gpu
            dm = RadioDataModule(cfg)
            if cfg.__dict__.get("gradient", False):
                model = GriGAN(cfg, args.exp_name, args.num_gpus)
            else:
                model = riGAN(cfg, args.exp_name, args.num_gpus)
        else:
            print("No valid experience selected in config file. Options are 'mri', 'mass_mapping', 'radio'.")
            exit()

    wandb_logger = WandbLogger(
        project=cfg.experience,
        name=args.exp_name,
        log_model=os.environ.get('WANDB_MODE') != 'offline',  # offline: don't stage 20 x 1.5 GB checkpoints
        save_dir=cfg.checkpoint_dir + 'wandb'
    )
    # plain-text loss curves next to the checkpoints (wandb stays the first logger, used for images)
    csv_logger = CSVLogger(save_dir=cfg.checkpoint_dir + args.exp_name + '/', name='csv_logs')

    
    os.makedirs(cfg.checkpoint_dir + args.exp_name + '/', exist_ok=True)
    class FromEpochCheckpoint(ModelCheckpoint):
        # --ckpt-from-epoch N: write no per-epoch checkpoints before epoch N (validation only uses epochs 80-99)
        def _should_skip_saving_checkpoint(self, trainer):
            return trainer.current_epoch < args.ckpt_from_epoch or super()._should_skip_saving_checkpoint(trainer)

    checkpoint_callback_epoch = FromEpochCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=cfg.checkpoint_dir + args.exp_name + '/',
        filename='checkpoint-{epoch}',
        every_n_epochs=1,
        save_top_k=20
    )

    try:
        accumulate_grad_batches = cfg.accumulate_grad_batches
    except: 
        accumulate_grad_batches = 1

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        strategy='ddp',
        max_epochs=cfg.num_epochs,
        callbacks=[checkpoint_callback_epoch],
        num_sanity_val_steps=2,
        profiler="simple",
        logger=[wandb_logger, csv_logger],
        benchmark=False,
        log_every_n_steps=10,
        accumulate_grad_batches=accumulate_grad_batches
    )

    if args.resume:
        trainer.fit(model, dm,
                    ckpt_path=cfg.checkpoint_dir + args.exp_name + f'/checkpoint-epoch={args.resume_epoch}.ckpt')
    else:
        trainer.fit(model, dm)