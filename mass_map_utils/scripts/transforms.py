"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Tuple, Union

import numpy as np
import torch

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()

def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """

    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std

def normalise_complex(
    shear: torch.Tensor, #Shape (2, H, W) 
    mag_mean: float = 0.14049194898307577,
    mag_std: float = 0.11606233247891737,
    eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalise a complex tensor. (Used during data transform)

    More specifically, we separate the magnitude and phase of the complex tensor, before
    normalising the magnitude given a mean and standard deviation of that magnitude. The default values
    are the mean and standard deviation across the entire dataset of mock shear maps, computed
    during preprocessing. 
    Once the magnitude is normalised, we re-integrate the phase, to returned a normalied complex tensor, with mean 
    and standard deviation of the magnitude.

    Args:
        shear: Input complex tensor to be normalised.
        mag_mean: Mean value of the magnitude of the complex tensor.
        mag_std: Standard deviation of the magnitude of the complex tensor.
        eps: Added to stddev to prevent dividing by zero.
    
    Returns:
        torch.Tensor: Normalised complex tensor
        float: mag_mean
        float: mag_std
    """

    magnitude = torch.abs(torch.complex(shear[0,:,:], shear[1,:,:]))
    phase = torch.angle(torch.complex(shear[0,:,:], shear[1,:,:])) #In radians

    normal_mag = (magnitude - mag_mean) / (mag_std + eps)
    normal_shear = normal_mag * torch.exp(1j*phase)
    normal_real = normal_shear.real
    normal_imag = normal_shear.imag
    return torch.stack((normal_real, normal_imag)), mag_mean, mag_std

def unnormalize_complex(
    normed_data: torch.Tensor, 
    mag_mean: float = 0.14049194898307577, 
    mag_std: float = 0.11606233247891737,
):  
    """
    Unnormalise a complex tensor.

    Tensors are normalised before being passed through the GAN, therefore this function 'unnormalises' the 
    output, to return a tensor with the same scale as the input. The magnitude and phase of the complex tensor
    are separated, and then the magnitude is unnormalised according to the mean and standard deviation of the shear. 
    Default values for these were calculated across the entire dataset of mock shear maps during preprocessing.
    Then, the phase is recombined with the unnormalised magnitude to return the unnormalised complex tensor.

    Args:
        normed_data: Normalised complex tensor to be unnormalised.
        mag_mean: Mean value of the magnitude of the complex tensor.
        mag_std: Standard deviation of the magnitude of the complex tensor.
    
    """
    # Sizes of tensors based on input to validate.py script, as that's where this function is called.
    normed_mag = torch.abs(torch.complex(normed_data[:,:,:,0], normed_data[:,:,:,1]))
    phase = torch.angle(torch.complex(normed_data[:,:,:,0], normed_data[:,:,:,1]))

    unnormed_mag = (normed_mag * mag_std) + mag_mean
    unnormed_data = unnormed_mag * torch.exp(1j*phase)
    unnormed_data_real = unnormed_data.real
    unnormed_data_imag = unnormed_data.imag

    #Permuted so output matches input shape: [batch size, img_size, img_size, re/imag]
    return torch.stack((unnormed_data_real, unnormed_data_imag)).permute(1,2,3,0) 
