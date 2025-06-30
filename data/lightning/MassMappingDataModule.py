import numpy as np
from torch.utils.data import DataLoader
from utils.parse_args import create_arg_parser
import pytorch_lightning as pl
from typing import Optional
from data.datasets.MM_data import (
    MassMappingDataset_Test,
    MassMappingDataset_Train,
    MassMappingDataset_Val,
    MassMappingDataset_Real
)
from utils.mri.transforms import to_tensor, normalize, normalize_instance, normalise_complex
from typing import Tuple
import pathlib
import torch


class MMDataTransform:
    def __init__(self, args, test=False, real_data=False, theta=5.0, ngal=30):
        self.args = args
        self.test = test
        self.real_data = real_data # where there is no gt
        self.theta = theta
        self.im_size = args.im_size
        self.ngal = ngal
        self.mask = None
        self.std1 = None
        self.std2 = None
        self.D = self.compute_fourier_kernel(self.im_size)


        # Load mask and std dev for noise
        try:
            if self.args.cosmo_dir_path is not None:
                self.mask = np.load(
                    self.args.cosmo_dir_path + "cosmos_mask.npy", allow_pickle=True
                ).astype(bool)
                self.std1 = np.load(
                    self.args.cosmo_dir_path + "cosmos_std1.npy", allow_pickle=True
                )
                self.std2 = np.load(
                    self.args.cosmo_dir_path + "cosmos_std2.npy", allow_pickle=True
                )
        except:
            print("There is a problem with the mask loading. Proceeding without mask.")
            self.args.cosmo_dir_path = None
            self.mask = None
            self.std1 = None
            self.std2 = None

    @staticmethod
    def compute_fourier_kernel(N: int) -> np.ndarray:
        """Computes the Fourier space kernel which represents the mapping between
            convergence (kappa) and shear (gamma).

        Args:
            N (int): x,y dimension of image patch (assumes square images).

        Returns:
            D (np.ndarray): Fourier space Kaiser-Squires kernel, with shape = [N,N].
        """
        # Generate grid of Fourier domain
        kx = np.arange(N).astype(np.float64) - N / 2
        kx, ky = np.meshgrid(kx, kx)
        k = kx**2 + ky**2
        # Define Kaiser-Squires kernel
        D = np.zeros((N, N), dtype=np.complex128)
        # Avoid divide by zero warning
        D[k > 0] = ((kx**2.0 - ky**2.0) + 1j * (2.0 * kx * ky))[k > 0] / k[k > 0]
        # Apply inverse FFT shift
        return np.fft.ifftshift(D)

    @staticmethod
    def forward_model(kappa: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Applies the forward mapping between convergence and shear through their
            relationship in Fourier space.

        Args:
            kappa (np.ndarray): (Real-valued) convergence field, with shape [N,N].
            D (np.ndarray): Fourier space Kaiser-Squires kernel, with shape = [N,N].

        Returns:
            gamma (np.ndarray): Shearing field, with shape [N,N].
        """
        F_kappa = np.fft.fft2(kappa)  # Perform 2D forward FFT
        F_gamma = F_kappa * D  # Map convergence onto shear
        return np.fft.ifft2(F_gamma)  # Perform 2D inverse FFT

    @staticmethod
    def backward_model(gamma: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Applies the backward mapping between shear and convergence through their
            relationship in Fourier space.

        Args:
            gamma (np.ndarray): Shearing field, with shape [N,N].
            D (np.ndarray): Fourier space Kaiser-Squires kernel, with shape = [N,N].

        Returns:
            kappa (np.ndarray): Convergence field, with shape [N,N].
        """
        F_gamma = np.fft.fft2(gamma)
        F_kappa = F_gamma / D
        F_kappa = np.nan_to_num(F_kappa, nan=0, posinf=0, neginf=0)
        return np.fft.ifft2(F_kappa)  # Allows for complex kappa here

    @staticmethod
    def noise_maker(
        theta: float, im_size: int, ngal: int, kappa: np.ndarray, D: np.ndarray
    ) -> np.ndarray:
        """Adds some random Gaussian noise to a mock weak lensing map.

        Args:
            theta (float): Opening angle in deg.
            im_size (int): Size of weak lensing map, in pixels.
            ngal (int): Number of galaxies.
            kappa (np.ndarray): Convergence map.
            D (np.ndarray): Precomputed Fourier kernel.

        Returns:
            gamma (np.ndarray): A synthetic representation of the shear field, gamma, with added noise.
        """
        sigma = 0.37 / np.sqrt(((theta * 60 / im_size) ** 2) * ngal)
        gamma = MMDataTransform.forward_model(kappa, D) + (sigma / np.sqrt(2)) * (
            np.random.randn(im_size, im_size) + 1j * np.random.randn(im_size, im_size)
        )
        return gamma

    def realistic_noise_maker(self, kappa: np.ndarray) -> np.ndarray:
        """Adds realistic simulated Gaussian noise to a mock weak lensing map.

        Args:
            im_size (int): Size of weak lensing map, in pixels.
            kappa (np.ndarray): Convergence map.

        Returns:
            gamma (np.ndarray): A synthetic representation of the shear field, gamma, with added noise.
        """
        gamma = MMDataTransform.forward_model(kappa, self.D) + (
            self.std1 * np.random.randn(self.im_size, self.im_size)
            + 1.0j * self.std2 * np.random.randn(self.im_size, self.im_size)
        )
        return gamma

    def gamma_gen(self, kappa: np.ndarray) -> np.ndarray:
        """Apply the forward model with the correct set of parameters.

        This function takes the input, kappa, applies the forward relationship to generate the corresponding value of the shear field,
        then adds Gaussian noise, to simulate a real observation.

        Args:
            kappa (np.ndarray): Convergence map/ground truth.

        Returns:
            gamma (np.ndarray): A synthetic representation of the shear field, gamma, with added noise.
        """
        if self.args.cosmo_dir_path is not None:
            return self.realistic_noise_maker(kappa)
        else:
            return MMDataTransform.noise_maker(
                self.theta, self.im_size, self.ngal, kappa, self.D
            )

    def __call__(self, data: np.ndarray) -> Tuple[float, float, float, float]:
        """Transforms the data.

        Note: gt = ground truth. The ground truth is the original kappa simulation from kappaTNG.
        Gamma represents the observation.

        Args:
            data (np.ndarray): Real-valued array, that is either real data (gamma) or simulations (kappa) during training/validation/testing.

        Returns:
            (tuple) tuple containing:
                normalized_gamma (float): Normalised measurement/gamma.
                normalized_gt (float): Normalised ground truth/kappa.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.

        """
        if self.real_data:
            gamma = data
            kappa = None
        else:
            # Generate observation on the fly.
            gamma = self.gamma_gen(
                data
            )  # real kappa here, but gamma_gen adds empty axis so gamma can be complex
            kappa = data

        ks = self.backward_model(gamma, self.D)

        # Format observation data.
        pt_gamma = to_tensor(gamma)  # Shape (H, W, 2)
        pt_gamma = pt_gamma.permute(2, 0, 1)  # Shape (2, H, W)

        # Pseudo-reconstruction of convergence field.
        pt_ks = to_tensor(ks)  # Shape (H, W, 2)
        pt_ks = pt_ks.permute(2, 0, 1)  # Shape (2, H, W)

        # Format input gt data.
        if not self.real_data:
            kappa = np.expand_dims(kappa, axis=-1)
            pt_gt = torch.from_numpy(kappa)# Shape (H, W, 1)
            pt_gt = pt_gt.permute(2, 0, 1)  # Shape (1, H, W)
            normalized_gt = normalize(
                pt_gt, 0.00015744006243248638, 0.02968584954283938
            ).float()  # Shape (1, H, W)
        else:
            normalized_gt = torch.empty(1, 0, 0)

        # Normalization step.
        normalized_gamma, mean, std = normalise_complex(pt_gamma)
        normalized_ks, mean_ks, std_ks = normalize_instance(pt_ks)
        normalized_ks = normalize(pt_ks, mean_ks, std_ks)

        normalized_gamma = torch.cat([normalized_gamma, normalized_ks], dim=0)

        # Mask the shear gamma
        if self.mask is not None:
            normalized_gamma[:, self.mask == 0] = 0.0

        # Return normalized measurements, normalized gt, mean, and std.xs
        return normalized_gamma.float(), normalized_gt, mean, std

class MMDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project.
    """

    def __init__(self, args):
        """The 'args' come from the config.yml file. See the docs for further information."""
        super().__init__()
        self.prepare_data_per_node = True
        self.args = args

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        train_data = MassMappingDataset_Train(
            data_dir=pathlib.Path(self.args.data_path) / "kappa_train",
            transform=MMDataTransform(self.args, test=False),
        )

        dev_data = MassMappingDataset_Val(
            data_dir=pathlib.Path(self.args.data_path) / "kappa_val",
            transform=MMDataTransform(self.args, test=True),
        )
        
        real_data = MassMappingDataset_Real(data_dir=pathlib.Path(self.args.data_path) / "cosmos",
        transform=MMDataTransform(self.args, test=True, real_data=True),)
        
        test_data = MassMappingDataset_Test(
            data_dir=pathlib.Path(self.args.data_path) / "kappa_test",
            transform=MMDataTransform(self.args, test=True),
            )

        self.train, self.validate, self.test, self.real = train_data, dev_data, test_data, real_data

    # define your dataloaders
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validate,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            drop_last=False,
        )
    
    def real_dataloader(self):
        return DataLoader(
            dataset = self.real,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            drop_last=False,
        )