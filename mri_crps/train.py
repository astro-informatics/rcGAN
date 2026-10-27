"""Train rcGAN-MRI with the rcGAN regulariser (R), fair CRPS (F) or multiscale fair CRPS (MS).

Same as the repo's train.py for the MRI experience, except: cached data module, CSV logging,
the regulariser switch, an optional limit on steps (smoke tests), and automatic resume from
the last checkpoint (so Slurm requeues continue the run).

Usage: python mri_crps/train.py --config configs/mri_crps.yml --exp-name R --reg rcgan
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.lightning.rcGAN import rcGAN  # noqa: E402
from mri_crps.data import CachedMRIDataModule  # noqa: E402


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/mri_crps.yml')
    p.add_argument('--exp-name', required=True)
    p.add_argument('--reg', default='rcgan', choices=['rcgan', 'fair', 'ms', 'plugin'])
    p.add_argument('--adv-scale', type=float, default=1.0)
    p.add_argument('--num-gpus', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=-1, help='smoke tests only')
    p.add_argument('--limit-val-batches', type=float, default=1.0, help='smoke tests only')
    p.add_argument('--max-time', default=None, help='DD:HH:MM:SS safety limit')
    args = p.parse_args()

    torch.set_float32_matmul_precision('medium')
    seed_everything(args.seed, workers=True)

    with open(args.config) as f:
        cfg = json.loads(json.dumps(yaml.load(f, Loader=yaml.FullLoader)), object_hook=load_object)

    dm = CachedMRIDataModule(cfg)
    model = rcGAN(cfg, args.exp_name, args.num_gpus, reg=args.reg, adv_scale=args.adv_scale)

    ckpt_dir = os.path.join(cfg.checkpoint_dir, args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = CSVLogger(save_dir=os.path.join(cfg.checkpoint_dir, 'logs'), name=args.exp_name, version=0)
    checkpoint_callback = ModelCheckpoint(
        monitor='epoch', mode='max', dirpath=ckpt_dir, filename='checkpoint-{epoch}',
        every_n_epochs=1, save_top_k=cfg.keep_checkpoints, save_last=True,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.num_gpus,
        strategy='ddp',
        max_epochs=cfg.num_epochs,
        max_steps=args.max_steps,
        max_time=args.max_time,
        limit_val_batches=args.limit_val_batches,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=2,
        logger=logger,
        benchmark=False,
        log_every_n_steps=10,
    )

    last = os.path.join(ckpt_dir, 'last.ckpt')
    trainer.fit(model, dm, ckpt_path=last if os.path.exists(last) else None)
