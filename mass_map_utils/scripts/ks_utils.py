import numpy as np
from data.lightning.MassMappingDataModule import MMDataTransform
import torch


def backward_model(γ: np.ndarray, 𝒟: np.ndarray) -> np.ndarray:
    """Applies the backward mapping between shear and convergence through their
      relationship in Fourier space.
    Args:
      γ (np.ndarray): Shearing field, with shape [N,N].
      𝒟 (np.ndarray): Fourier space Kaiser-Squires kernel, with shape = [N,N].
    Returns:
      𝜅 (np.ndarray): Convergence field, with shape [N,N].
    """
    𝓕γ = np.fft.fft2(γ)  # Perform 2D forward FFT
    𝓕𝜅 = 𝓕γ / 𝒟  # Map convergence onto shear
    𝓕𝜅 = np.nan_to_num(𝓕𝜅, nan=0, posinf=0, neginf=0)  # Remove singularities
    return np.fft.ifft2(𝓕𝜅)  # Perform 2D inverse FFT


# def rmse(a: torch.float64, b: torch.float64, mask: np.ndarray = None) -> float:
#     """
#     args:
#         a (torch.float64): ground truth
#         b (torch.float64): reconstruction
#         mask (np.ndarray): Boolean mask
#     returns:
#         rmse (float): root mean squared error
#     """
#     if mask is not None:
#         a = a[mask == 1]
#         b = b[mask == 1]
#     return torch.sqrt(torch.mean(torch.square(a - b)))


# def pearsoncoeff(a: torch.float64, b: torch.float64, mask: np.ndarray = None) -> float:
#     """
#     args:
#         a (torch.float64): ground truth
#         b (torch.float64): reconstruction
#         mask (np.ndarray): mask
#     returns:
#         pearson (float): Pearson correlation coefficient
#     """
#     if mask is not None:
#         a = a[mask == 1]
#         b = b[mask == 1]
#     a -= torch.mean(a)
#     b -= torch.mean(b)
#     num = torch.sum(a * b)
#     denom = torch.sqrt(torch.sum(a**2) * torch.sum(b**2))
#     return num / denom


# def psnr(a: torch.float64, b: torch.float64, mask: np.ndarray = None) -> float:
#     """
#     args:
#         a (torch.float64): ground truth
#         b (torch.float64): reconstruction
#         mask (np.ndarray): mask
#     returns:
#         psnr (float): peak signal-to-noise ratio
#     """
#     if mask is not None:
#         a = a[mask == 1]
#         b = b[mask == 1]
#     mse = torch.mean((a - b) ** 2)
#     r = a.max()
#     return 10 * torch.log10(r / mse)


# def snr(a: torch.float64, b: torch.float64, mask: np.ndarray = None) -> float:
#     """
#     args:
#         a (torch.float64): ground truth
#         b (torch.float64): reconstruction
#         mask (np.ndarray): mask
#     returns:
#         snr (float): signal-to-noise ratio
#     """
#     if mask is not None:
#         a = a[mask == 1]
#         b = b[mask == 1]
#     signal = torch.mean(a**2)
#     noise = torch.mean((a - b) ** 2)
#     return 10 * torch.log10(signal / noise)

# NUMPY VERSIONS


def rmse(a: np.ndarray, b: np.ndarray, mask: bool) -> float:
    """
    args:
        a (np.ndarray): ground truth
        b (np.ndarray): reconstruction
        mask (bool): mask
    returns:
        rmse (float): root mean squared error
    """
    if a.shape != b.shape:
        print(f"Shape of a: {a.shape}, Shape of b: {b.shape}")
        raise ValueError("Shapes of a and b do not match.")

    a = a[mask == 1]
    b = b[mask == 1]
    return np.sqrt(np.mean(np.square(a - b)))


def pearsoncoeff(a: np.ndarray, b: np.ndarray, mask: bool) -> float:
    """
    args:
        a (np.ndarray): ground truth
        b (np.ndarray): reconstruction
        mask (bool): mask
    returns:
        pearson (float): Pearson correlation coefficient
    """
    if a.shape != b.shape:
        print(f"Shape of a: {a.shape}, Shape of b: {b.shape}")
        raise ValueError("Shapes of a and b do not match.")

    a = a[mask == 1]
    b = b[mask == 1]
    a -= np.mean(a)
    b -= np.mean(b)
    num = np.sum(a * b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return num / denom


def psnr(a: np.ndarray, b: np.ndarray, mask: bool) -> float:
    """
    args:
        a (np.ndarray): ground truth
        b (np.ndarray): reconstruction
        mask (bool): mask
    returns:
        psnr (float): peak signal-to-noise ratio
    """
    if a.shape != b.shape:
        print(f"Shape of a: {a.shape}, Shape of b: {b.shape}")
        raise ValueError("Shapes of a and b do not match.")

    a = a[mask == 1]
    b = b[mask == 1]
    mse = np.mean((a - b) ** 2)
    r = a.max()
    return 10 * np.log10((r ** 2) / mse)


def snr(a: np.ndarray, b: np.ndarray, mask: bool) -> float:
    """
    args:
        a (np.ndarray): ground truth
        b (np.ndarray): reconstruction
        mask (bool): mask
    returns:
        snr (float): signal-to-noise ratio
    """
    if a.shape != b.shape:
        print(f"Shape of a: {a.shape}, Shape of b: {b.shape}")
        raise ValueError("Shapes of a and b do not match.")

    a = a[mask == 1]
    b = b[mask == 1]
    signal = np.mean(a**2)
    noise = np.mean((a - b) ** 2)
    return 10 * np.log10(signal / noise)

# def ecp(samples, gt, lower_quant=5, upper_quant=5):
#     lower = np.percentile(samples, lower_quant, axis=0)
#     upper = np.percentile(samples, upper_quant, axis=0)
#     inside_interval = (gt >= lower) & (gt <= upper)
#     return np.mean(inside_interval)
