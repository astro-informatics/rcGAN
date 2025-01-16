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