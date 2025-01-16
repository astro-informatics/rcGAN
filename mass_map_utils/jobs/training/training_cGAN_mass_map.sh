#!/bin/bash

#SBATCH --job-name=training
#SBATCH -p GPU
# requesting one node
# SBATCH -N1
# requesting 12 cpus
# SBATCH -n12
#SBATCH --ntasks=4                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=4          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --cpus-per-task=16           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH --gres=gpu:a100:4            # requesting GPUs
#SBATCH --mem-per-gpu=80GB           # memory per GPU
#SBATCH --mail-use=jessica.whitney.22@ucl.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=tr_real_output_%j.out
#SBATCH --error=tr_real_output_%j.err



# Load conda and activate environment
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN

export WANDB_DIR=/share/gpu0/jjwhit/wandb/logs
export WANDB_CACHE_DIR=/share/gpu0/jjwhit/wandb/.cache/wandb
export WANDB_CONFIG_DIR=/share/gpu0/jjwhit/wandb/.config/wandb

# Echo commands
set -x

echo $CUDA_VISIBLE_DEVICES
echo $WANDB_DIR
echo $WANDB_CACHE_DIR
echo $WANDB_CONFIG_DIR

cd /home/jjwhit/rcGAN

srun python -u train.py --config ./configs/mass_map.yml --exp-name mmgan_training_real_output --num-gpus 4
