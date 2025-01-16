#!/bin/bash

#SBATCH --job-name=find_batch_size
#SBATCH -p GPU
#SBATCH --nodelist=compute-gpu-0-2
# requesting one node, one gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1    # requesting GPUs
#SBATCH --ntasks-per-node=1
#SBATCH --output=find_batch_size_%j.out
#SBATCH --error=find_batch_size_%j.err

# Finding batch size only on one GPU
# Make sure to specifiy the node and/or kind of GPU you want to train the full model on.

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

# make sure to run on only 1 GPU
srun python -u  find_batch_size.py --config ./configs/radio_image_test.yml --num-gpus 1


