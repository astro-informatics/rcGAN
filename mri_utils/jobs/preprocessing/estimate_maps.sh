#!/bin/bash

#SBATCH --job-name=estimate_maps 
#SBATCH -p GPU
# requesting one node
# SBATCH -N1
# requesting 12 cpus
# SBATCH -n12
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --cpus-per-task=12           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH --gres=gpu:v100:4            # requesting GPUs
#SBATCH --mail-use=t.liaudat@ucl.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=estimate_maps_%j.out
#SBATCH --error=estimate_maps_%j.err



# Load conda and activate environment
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN

# Echo commands
set -x

echo $CUDA_VISIBLE_DEVICES

cd /home/tl3/repos/project-rcGAN/rcGAN

srun python -u scripts/mri/estimate_maps.py  --sense-maps-val
srun python -u scripts/mri/estimate_maps.py

