import numpy as np

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
    if a.shape != b.shape:
        print(f"Shape of a: {a.shape}, Shape of b: {b.shape}")
        raise ValueError("Shapes of a and b do not match.")

    a = a[mask == 1]
    b = b[mask == 1]
    signal = np.mean(a**2)
    noise = np.mean((a - b) ** 2)
    return 10 * np.log10(signal / noise)
