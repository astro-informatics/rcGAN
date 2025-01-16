import numpy as np
import os
import sys
sys.path.append("/home/jjwhit/rcGAN")
import yaml
import json
import types
from utils.parse_args import create_arg_parser
from data.lightning.MassMappingDataModule import MMDataTransform as MM

def load_object(dct):
    return types.SimpleNamespace(**dct)
args = create_arg_parser().parse_args()
with open(args.config, "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = json.loads(json.dumps(cfg), object_hook=load_object)

data_dir = cfg.data_path + 'kappa_train/'

files = os.listdir(data_dir)
kappa_mean=[]
shear_mean=[]

im_size=cfg.im_size

mask =  np.load(
    cfg.cosmos_dir_path + 'cosmos_mask.npy', allow_pickle=True
).astype(bool)
std1 = np.load(
    cfg.cosmos_dir_path + 'cosmos_std1.npy', allow_pickle=True
)
std2 = np.load(
    cfg.cosmos_dir_path + 'cosmos_std2.npy', allow_pickle=True
)
kernel = MM.compute_fourier_kernel(im_size)

count = 0
for file in files:
    data = np.load(os.path.join(data_dir, file))
    kappa_mean.append(np.mean(data))
    shear = MM.forward_model(data, kernel)
    noisy_shear = shear + (
            std1 * np.random.randn(im_size, im_size) 
            + 1.j * std2 * np.random.randn(im_size, im_size)
    )
    #Decompose shear into magnitude and phase
    magnitude = np.abs(noisy_shear)
    #Compute mean of magnitude

    shear_mean.append(np.mean(magnitude))
    count += 1
    if count >= 500:
        break

total_kappa_mean = np.mean(kappa_mean)
print('Total kappa mean is: ', total_kappa_mean)
total_shear_mean = np.mean(shear_mean)
print('Total shear magnitude mean is: ', total_shear_mean)

count=0
std_i = 0
std_k = 0
for file in files:
    data = np.load(os.path.join(data_dir, file))
    kappa_i = (data - total_kappa_mean)**2
    std_i += np.sum(kappa_i)
    shear = MM.forward_model(data, kernel)
    noisy_shear = shear + (
            std1 * np.random.randn(im_size, im_size) 
            + 1.j * std2 * np.random.randn(im_size, im_size) 
    )
    #Decompose shear into magnitude and phase
    magnitude = np.abs(noisy_shear)
    shear_i = (magnitude - total_shear_mean)**2
    std_k += np.sum(shear_i)
    count += 1
    if count >= 500:
        break

total_std_kappa = np.sqrt(std_i/(count*im_size**2))
print('Kappa standard deviation is: ', total_std_kappa)
total_std_shear = np.sqrt(std_k/(count*im_size**2))
print('Shear mean standard deviation is: ', total_std_shear)