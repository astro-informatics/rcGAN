#!/bin/bash

#SBATCH --job-name=tr_image_radio
#SBATCH -p GPU
#SBATCH --nodelist=compute-gpu-0-2
# requesting one node
#SBATCH --nodes=1
##SBATCH --gres=gpu:a100:4            # requesting GPUs
#SBATCH --gres=gpu:4    # requesting GPUs
#SBATCH --ntasks-per-node=4          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
##SBATCH --mem-per-gpu=80GB           # memory per GPU
#SBATCH --output=tr_radio_image_%j.out
#SBATCH --error=tr_radio_image_%j.err

##SBATCH -n12
##SBATCH --cpus-per-task=4           # nombre de coeurs CPU par tache (un quart du noeud ici)
## SBATCH --gres=gpu:1          # requesting GPUs
#SBATCH --mail-use=matthijs.mars.20@ucl.ac.uk
#SBATCH --mail-type=ALL


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

# srun python -u  train.py --radio_image --exp-name radio_train_varying --num-gpus 4
# python scripts/radio/validate.py --exp-name radio_train_varying
python scripts/radio/plot.py --exp-name radio_train_varying
