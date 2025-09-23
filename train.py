import torch
import yaml
import types
import json
import sys
import os

sys.path.append(
    "/lustre/fswork/projects/rech/rbn/ulx23va/projects/unrolled_cGAN/repos/rcGAN/"
)

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

# from data.lightning.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from models.lightning.rcGAN import rcGAN
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from data.lightning.MassMappingDataModule import MMDataModule
from data.lightning.RadioDataModule import RadioDataModule
from models.lightning.mmGAN import mmGAN
from models.lightning.riGAN import riGAN
from models.lightning.GriGAN import GriGAN


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)

    print(f"Experiment Name: {args.exp_name}")
    print(f"Number of GPUs: {args.num_gpus}")
    print("Device count: ", torch.cuda.device_count())
    print(f"Config file path: {args.config}")

    config_path = args.config

    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        # Set WANDB to offline mode if specified in config
        wandb_offline = getattr(cfg, "wandb_offline", True)
        log_model = True
        if wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
            # Disable model saving in offline mode, as it is not supported
            wandb_log_model = False

        # Load the correct model
        if cfg.experience == "mri":
            dm = MRIDataModule(cfg)
            model = rcGAN(cfg, args.exp_name, args.num_gpus)
        elif cfg.experience == "mass_mapping":
            dm = MMDataModule(cfg)
            model = mmGAN(cfg, args.exp_name, args.num_gpus)
        elif cfg.experience == "radio":
            cfg.num_workers = args.num_gpus  # set number of workers to same as gpu
            dm = RadioDataModule(cfg)
            if cfg.__dict__.get("gradient", False):
                model = GriGAN(cfg, args.exp_name, args.num_gpus)
            else:
                model = riGAN(cfg, args.exp_name, args.num_gpus)
        else:
            print(
                "No valid experience selected in config file. Options are 'mri', 'mass_mapping', 'radio'."
            )
            exit()

    wandb_logger = WandbLogger(
        project=cfg.experience,
        name=args.exp_name,
        log_model=wandb_log_model,
        save_dir=cfg.checkpoint_dir + "wandb",
        offline=wandb_offline,
    )

    os.makedirs(cfg.checkpoint_dir + args.exp_name + "/", exist_ok=True)
    checkpoint_callback_epoch = ModelCheckpoint(
        monitor="epoch",
        mode="max",
        dirpath=cfg.checkpoint_dir + args.exp_name + "/",
        filename="checkpoint-{epoch}",
        every_n_epochs=1,
        save_top_k=20,
        save_last=True,  # Always save the last checkpoint
    )

    try:
        accumulate_grad_batches = cfg.accumulate_grad_batches
    except:
        accumulate_grad_batches = 1

    # Configure DDP strategy with find_unused_parameters to avoid port conflicts
    from pytorch_lightning.strategies import DDPStrategy
    import socket

    def find_free_port():
        """Find a free port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    # Set a free port for distributed training
    free_port = find_free_port()
    os.environ["MASTER_PORT"] = str(free_port)
    os.environ["MASTER_ADDR"] = "localhost"
    print(f"Using MASTER_PORT: {free_port}")

    # Configure DDP strategy for better GPU utilization
    ddp_strategy = DDPStrategy(
        find_unused_parameters=False,  # Set to False for better performance
        process_group_backend="nccl",
        static_graph=True,  # Enable static graph optimization
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=ddp_strategy,
        max_epochs=cfg.num_epochs,
        callbacks=[checkpoint_callback_epoch],
        num_sanity_val_steps=2,
        profiler="simple",
        logger=wandb_logger,
        benchmark=False,
        log_every_n_steps=10,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    if args.resume:
        trainer.fit(
            model,
            dm,
            ckpt_path=cfg.checkpoint_dir
            + args.exp_name
            + f"/checkpoint-epoch={args.resume_epoch}.ckpt",
        )
    else:
        trainer.fit(model, dm)
