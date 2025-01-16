#!/bin/bash

#SBATCH --job-name=multi_gpu_train
#SBATCH -p GPU

# request specific node
#SBATCH --nodelist=compute-gpu-0-2
# requesting one node, GPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:4          # n_gpus
#SBATCH --ntasks-per-node=4   # ntasks needs to be same as n_gpus

#SBATCH --output=multi_gpu_train_%j.out
#SBATCH --error=multi_gpu_train_%j.err

#SBATCH -n16
#SBATCH --cpus-per-task=4       # 4 cpus per task

##SBATCH --mail-use=
##SBATCH --mail-type=ALL


# Load conda and activate environment
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN

export WANDB_DIR=/share/gpu0/mars/wandb/logs
export WANDB_CACHE_DIR=/share/gpu0/mars/wandb/.cache/wandb
export WANDB_CONFIG_DIR=/share/gpu0/mars/wandb/.config/wandb

# Echo commands
set -x

echo $CUDA_VISIBLE_DEVICES
echo $WANDB_DIR
echo $WANDB_CACHE_DIR
echo $WANDB_CONFIG_DIR

cd /home/mars/git/rcGAN

srun python -u  train.py --config ./configs/radio_image_test.yml --exp-name radio_train_test --num-gpus 4

