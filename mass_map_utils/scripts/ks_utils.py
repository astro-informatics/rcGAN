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


# def Gaussian_smoothing(kappa:np.ndarray,n:int,theta:float,epsilon=25) -> np.ndarray:
#     """Applies Gaussian smoothing to a convergence map.

#     This is done by taking Fourier transform of the convergence map, and a Gaussian,
#     convolving them, and then applying an inverse Fourier transform to the result.

#     Args:
#         kappa (np.ndarray): Convergence map.
#         n (int): The dimensions, in pixels, of a square map kappa, where n x n is the no. of pixels in kappa.
#         theta (float): Opening angle in deg.
#         epsilon (int): Smoothing scale.

#     Returns:
#         smoothed_kappa (np.ndarray): Returns a smoothed representation of the the convergence field.
#     """
#     kappa_f = np.fft.fft2(kappa) #Fourier transform of kappa
#     kappa_f_shifted = np.fft.fftshift(kappa_f) #Changes the indexing of the Fourier coefficients

#     Gaussian_filter = np.zeros((n,n))
#     i = (epsilon*n)/(60*theta)
#     sig_pix = i/(2*np.sqrt(2*np.log(2)))

#     t = int(n/2)
#     y = np.arange(-t,t)
#     xx,yy = np.meshgrid(y,y)

#     exponential = np.exp(-(xx**2 + yy**2)*2*(sig_pix*np.pi)**2)
#     Gaussian_filter = exponential

#     smoothed_kappa_f = np.fft.ifftshift(kappa_f_shifted*Gaussian_filter)
#     smoothed_kappa = np.fft.ifft2(smoothed_kappa_f)
#     return smoothed_kappa


"""
ks93 and ks93inv from lenspack (Austin Peel, Cosmostat).
https://github.com/austinpeel/lenspack/blob/master/lenspack/image/inversion.py
"""


def ks93(g1, g2):
    """Direct inversion of weak-lensing shear to convergence.

    This function is an implementation of the Kaiser & Squires (1993) mass
    mapping algorithm. Due to the mass sheet degeneracy, the convergence is
    recovered only up to an overall additive constant. It is chosen here to
    produce output maps of mean zero. The inversion is performed in Fourier
    space for speed.

    Parameters
    ----------
    g1, g2 : array_like
        2D input arrays corresponding to the first and second (i.e., real and
        imaginary) components of shear, binned spatially to a regular grid.

    Returns
    -------
    kE, kB : tuple of numpy arrays
        E-mode and B-mode maps of convergence.

    Raises
    ------
    AssertionError
        For input arrays of different sizes.
    """
    # Check consistency of input maps
    assert g1.shape == g2.shape

    # Compute Fourier space grids
    (nx, ny) = g1.shape
    k1, k2 = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))

    # Compute Fourier transforms of g1 and g2
    g1hat = np.fft.fft2(g1)
    g2hat = np.fft.fft2(g2)

    # Apply Fourier space inversion operator
    p1 = k1 * k1 - k2 * k2
    p2 = 2 * k1 * k2
    k2 = k1 * k1 + k2 * k2
    k2[0, 0] = 1  # avoid division by 0
    kEhat = (p1 * g1hat + p2 * g2hat) / k2
    kBhat = -(p2 * g1hat - p1 * g2hat) / k2

    # Transform back to pixel space
    kE = np.fft.ifft2(kEhat).real
    kB = np.fft.ifft2(kBhat).real

    return kE, kB


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

#NUMPY VERSIONS

def rmse(a: np.ndarray, b: np.ndarray, mask: bool) -> float:
    """
    args:
        a (np.ndarray): ground truth
        b (np.ndarray): reconstruction
        mask (bool): mask
    returns:
        rmse (float): root mean squared error
    """
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
    a = a[mask == 1]
    b = b[mask == 1]
    mse = np.mean((a - b) ** 2)
    r = a.max()
    return 10 * np.log10(r / mse)


def snr(a: np.ndarray, b: np.ndarray, mask: bool) -> float:
    """
    args:
        a (np.ndarray): ground truth
        b (np.ndarray): reconstruction
        mask (bool): mask
    returns:
        snr (float): signal-to-noise ratio
    """
    a = a[mask == 1]
    b = b[mask == 1]
    signal = np.mean(a**2)
    noise = np.mean((a - b) ** 2)
    return 10 * np.log10(signal / noise)
