# rcGAN development version  


# Installation

If in a computing cluster, first run:
``` bash
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
```


First install the conda dependencies setting the correct channels:
``` bash
conda create --name cGAN --file conda_requirements.txt --channel pytorch --channel nvidia --channel conda-forge --channel defaults
```

Then activate the conda environment and install the pip requirements:
``` bash
conda activate cGAN
pip install -r pypi_requirements.txt
```

# Logging

## Weight and biases

Parameters and environmental variables
WANDB_CACHE_DIR
WANDB_DATA_DIR

logs -> `./wandb` -> `WANDB_DIR`
artifacts -> `~/.cache/wandb` -> `WANDB_CACHE_DIR`
configs -> `~/.config/wandb` -> `WANDB_CONFIG_DIR`

# Set the variables
``` bash
export WANDB_DIR=/share/gpu0/jjwhit/wandb/logs
export WANDB_CACHE_DIR=/share/gpu0/jjwhit/wandb/.cache/wandb
export WANDB_CONFIG_DIR=/share/gpu0/jjwhit/wandb/.config/wandb
```

# Training the model

Training is as simple as running the following command:
```python
python train.py --config ./configs/mass_map.yml --exp-name rcgan_test --num-gpus X
```
where ```X``` is the number of GPUs you plan to use. Note that this project uses Weights and Biases (wandb) for logging.
See [their documentation](https://docs.wandb.ai/quickstart) for instructions on how to setup environment variables.
Alternatively, you may use a different logger. See PyTorch Lightning's [documentation](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) for options.

If you need to resume training, use the following command:
```python
python train.py --config ./configs/mass_map.yml --exp-name rcgan_test --num-gpus X --resume --resume-epoch Y
```
where ```Y``` is the epoch to resume from.

By default, we save the previous 50 epochs. Ensure that your checkpoint path points to a location with sufficient disk space.
If disk space is a concern, 50 can be reduced to 25.
This is important for the next step, validation.


## Multi-GPU Runs
To make the lightning module work on multiple GPUs (and on multiple nodes) when using the SLURM workload manager, we need to be careful in setting up the SLURM job script. An example of how to do this can be found here https://pytorch-lightning.readthedocs.io/en/1.2.10/clouds/slurm.html. 

In particular if we want to run on 4 GPUs on one node we need to make sure that we ask for 4 GPUs as well as 4 tasks (since lightning will create 1 task per GPU) per node:

```
#SBATCH --gres=gpu:4          # n_gpus
#SBATCH --ntasks-per-node=4   # ntasks needs to be same as n_gpus
```

An example of a job-script for training using multiple GPUs can be found in [examples/example_multi_gpu.sh](https://github.com/astro-informatics/rcGAN/blob/dev-multiGPU/examples/example_multi_gpu_train.sh)

## Batch size tuning
Additionally I have created a script, [find_batch_size.py](https://github.com/astro-informatics/rcGAN/blob/dev-multiGPU/find_batch_size.py) that finds the largest batch_size that you can run per GPU. This depends on the VRAM available on the GPU and can therefore vary accross machines/nodes. An example job file can be found in [examples/example_find_batch_size.sh](https://github.com/astro-informatics/rcGAN/blob/dev-multiGPU/examples/example_find_batch_size.sh). Usage is:

```
python find_batch_size.py --config [config_file.yml]
```

Finally, to support larger batch sizes we can accumulate the gradients over batch sizes. In order to enable this and set the amount of accumulation you can add to your config file:

```
batch_size: 8               # batch_size per GPU (because of DDP)
accumulate_grad_batches: 2  # updates model after 2 batches per GPU
```

When using the distributed data processing (DDP) training strategy, the model is copied exactly on each GPU and they all see only a part of the data during the epoch. After processing 1 batch on each of the GPUs, the gradients from each of the GPUs are averaged and the models are updated. If we use gradient accumulation the gradients are instead averaged over several of such steps. The effective batch size of the model is therefore: n_gpus * batch_size *  accumulate_grad_batches. 