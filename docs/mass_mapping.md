# Reproducing our MRI results
## Update Config
Update ```configs/mass_map.yml``` with the path to your data, where you want to store checkpoints, and with the path to additional data such as masks.

## Data Setup
All of the scripts required are in the ```mass_map_utils``` folder. We include some preprocessing scripts, currently configured for the kappaTNG mock weak lensing maps.

## Training
Training is as simple as running the following command:
```python
python train.py --config ./configs/mass_map.yml --exp-name training_name --num-gpus X
```
where training_name will be used to access checkpoints for validation/testing/plotting, and for tracking weights and biases via wandb. ```X``` is the number of GPUs you plan to use. 

See wandb documentation (https://docs.wandb.ai/quickstart) for instructions on how to setup environment variables.
Alternatively, you may use a different logger. See PyTorch Lightning's [documentation](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) for options.

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

## Questions and Concerns
If you have any questions, or run into any issues, don't hesitate to reach out at jessica.whitney.22@ucl.ac.uk