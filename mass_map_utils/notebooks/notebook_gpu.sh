#!/bin/bash
#SBATCH --partition GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu 8G
##SBATCH --gres=gpu:v100:1
#SBATCH --gres=gpu:a100:1
#SBATCH --time 1-0:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --mail-use=ucasjjw@ucl.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output jupyter-notebook-%J.log
#Based on https://docs.ycrc.yale.edu/clusters-at-yale/guides/jupyter/
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

# print tunneling instructions jupyter-log
echo -e "To connect:
ssh -L ${port}:${node}:${port} ${user}@hypatia-login.hpc.phys.ucl.ac.uk

Use a Browser on your local machine to go to:
localhost:${port}

Remember to scancel job when done. Check output below for access token if
you need it.

Before running this jobs the jupyter lab needs to generate a cust password
Do this after activating the conda environment. REMEMBER THIS.
Run: "jupyter lab password" to set the password you'll need to acess the jupyter lab server
"

# source /share/apps/anaconda/3-2019.03/etc/profile.d/conda.sh
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN


# To run this job and put it ont he queue: sbatch notebook_gpu.sh

srun -n1 jupyter lab --no-browser --port=${port} --ip=${node}


