# Reproducing our mass-mapping results

This .md file has all the basic information required to reproduce the results from the [MMGAN paper](https://arxiv.org/abs/2410.24197).

## Setup Instructions
Install the required modules via anaconda:
```
conda create --name <env> --file requirements.txt
```
Note that you may need to change the PyTorch version, depending on your CUDA distribution.

## Update Config
Update ```configs/mass_map.yml``` with the path to your data, where you want to store checkpoints, and with the path to additional data such as masks.

## Data Setup
All of the scripts required are in the ```mass_map_utils``` folder. For our training we used the [kappaTNG mock weak lensing suite](https://columbialensing.github.io/#tng) and the [Schrabback et al.](https://www.aanda.org/articles/aa/abs/2010/08/aa13577-09/aa13577-09.html) COSMOS shape catalog to build a set of COSMOS-like convergence maps. To build the same dataset yourself run the ```mass_map_utils/jobs/preprocessing/create_dataset.sh``` bash file. This will run a series of scripts which will create the mock convergence maps, crop them down to 300x300 pixel maps, and then normalise the dataset. No shear maps are saved in the pre-processing step, they are all created on the fly by MMGAN during training.

If you wish to modify the dataset creation process, the python scripts themselves can be found in ```mass_map_utils/scripts```.

# Logging
By default, our model is tracked by Weights and Biases platform. See [their documentation](https://docs.wandb.ai/quickstart) for instructions on how to setup environment variables.
Alternatively, you may use a different logger. See PyTorch Lightning's [documentation](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) for options.

## Weight and biases

Parameters and environmental variables
WANDB_CACHE_DIR
WANDB_DATA_DIR
will need to be updated to your desired directory in the .sh files in ```mass_map_utils/jobs``` using the following command:
``` bash
export WANDB_DIR=<directory_name>/wandb/logs
export WANDB_CACHE_DIR=<directory_name>/wandb/.cache/wandb
export WANDB_CONFIG_DIR=<directory_name>/wandb/.config/wandb
```
where <directory_name> is the name of the directory you wish to save your logs to.

logs -> `./wandb` -> `WANDB_DIR`
artifacts -> `~/.cache/wandb` -> `WANDB_CACHE_DIR`
configs -> `~/.config/wandb` -> `WANDB_CONFIG_DIR`

## Training
Training is as simple as running the following command:
```python
python train.py --config ./configs/mass_map.yml --exp-name training_name --num-gpus X
```
where training_name will be used to access checkpoints for validation/testing/plotting, and for tracking weights and biases via wandb. ```X``` is the number of GPUs you plan to use. 

See wandb documentation (https://docs.wandb.ai/quickstart) for instructions on how to setup environment variables.
Alternatively, you may use a different logger. See PyTorch Lightning's [documentation](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) for options.

If you need to resume training at any point, use the following command:
```python
python train.py --config ./configs/mass_map.yml --exp-name training_name --num-gpus X --resume --resume-epoch Y
```
where ```Y``` is the epoch to resume from.

By default, we save the previous 50 epochs. Ensure that your checkpoint path points to a location with sufficient disk space.
If disk space is a concern, 50 can be reduced to 25.

For details specific to multi-GPU runs and batch size tuning please refer to ```docs/comments.md```.

## Validation
During training, validation is necessary in order to update the weight applied to
the standard deviation reward. There are a variety of metrics that can be assessed during validation, it is up to you what you'd rather use to choose the best training epoch. By default the model will select the model which the best RMSE results. Once completed, all other checkpoints will be automatically deleted this - to toggle this edit the end of the  ```validation.py``` file.

To validate, run the following command:
```python
python /scripts/mass_map/validate.py --config ./configs/mass_map.yml --exp-name mmgan_training_real_output
```

## Testing
To test the model's PSNR, SSIM, LPIPS, DISTS, and APSD, execute the following command:
```python
python /scripts/mass_map/test.py --config ./configs/mass_map.yml --exp-name mmgan_training_real_output
```
This will test all aforementioned metrics on the average reconstruction for 1, 2, 4, 8, 16, and 32 samples.

## Generating Posterior Samples
To generate figures similar to those found in our paper, execute the following command:
```python
python /scripts/mass_map/plot.py --config ./configs/mass_map.yml --exp-name mmgan_training_real_output --num-figs 10
```
where ```--num-figs``` controls the number of sets of samples to generate. This script will generate posterior samples, then save the associated reconstruction, uncertainty estimates, shear map, ground truth, and absolute error.
We use a jupyter notebooks ```mass_map_utils/notebooks/plot.ipynb``` to plot these samples.

To reproduce our cosmos results run the following command to generate samples:
```python /scripts/mass_map/cosmos_plot.py --config ./configs/mass_map.yml --exp-name mmgan_training_real_output
```
and plot via ```mass_map_utils/notebooks/cosmos_samples/ipynb```.

# General Mass-Mapping

If you would like to use MMGAN for your own mass-mapping problems there are a number of things to configure.

## Config

We chose to normalise across the entire dataset, rather than on an instance by instance case. Therefore we recommend, making a copy of ```configs/mass_map.yml```
with updated directory paths, as well as an updated kappa_mean and kappa_std configured to your dataset. 
We provide a script ```mass_map_utils/scripts/normalisation.py``` in order to calculate these values.

To find the optimal batch size, we have created a script in ```examples```, where the config file should be updated to a test version of the configuration 
you plan on using.

## Data

Currently, we provide two functions to add noise to the mock shear maps. We recommend using the realistic_noise_maker 
function found in ```data/lightning/MassMappingDataModule.py```, and replacing the standard deviation with that of
your dataset. Currently it is configured for the COSMOS data, with a standard deviation calculated using the COSMOS galaxy
catalog.


## Questions and Concerns
If you have any questions, or run into any issues, don't hesitate to reach out at jessica.whitney.22@ucl.ac.uk