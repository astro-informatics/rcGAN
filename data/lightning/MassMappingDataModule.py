import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional
from data.datasets.MM_data import MassMappingDataset_Test, MassMappingDataset_Train, MassMappingDataset_Val
from utils.mri import transforms
from typing import Tuple
import pathlib



class MMDataTransform:
    def __init__(self, args, test=False, theta=5.0, ng=384, ngal=30):
        self.args = args
        self.test =test
        self.theta = theta
        self.ng = ng
        self.ngal = ngal

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
        kx = np.arange(N).astype(np.float64) - N/2
        ky, kx = np.meshgrid(kx, kx)
        k = kx**2 + ky**2
        # Define Kaiser-Squires kernel
        D = np.zeros((N, N), dtype=np.complex128)
        # D = np.where(k > 0, ((kx ** 2.0 - ky ** 2.0) + 1j * (2.0 * kx * ky))/k, D)
        # Another formulation to avoid divide by zero warning
        D[k>0] = (((kx ** 2.0 - ky ** 2.0) + 1j * (2.0 * kx * ky))[k>0]/k[k>0])
        # Apply inverse FFT shift 
        return np.fft.ifftshift(D)

    @staticmethod
    def forward_model(kappa: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Applies the forward mapping between convergence and shear through their 
            relationship in Fourier space.

        Args:
            kappa (np.ndarray): Convergence field, with shape [N,N].
            D (np.ndarray): Fourier space Kaiser-Squires kernel, with shape = [N,N].

        Returns:
            gamma (np.ndarray): Shearing field, with shape [N,N].
        """
        F_kappa = np.fft.fft2(kappa) # Perform 2D forward FFT
        F_gamma = F_kappa * D # Map convergence onto shear
        return np.fft.ifft2(F_gamma) # Perform 2D inverse FFT

    @staticmethod
    def noise_maker(theta, ngrid, ngal, kappa) -> np.ndarray:
        """Adds some random Gaussian noise to a mock weak lensing map.

        Args:
            theta (float): Opening angle in deg.
            ngrid (int): Number of grids.
            ngal (int): Number of galaxies.
            kappa (np.ndarray): Convergence map.
        
        Returns:
            gamma (np.ndarray): A synthetic representation of the shear field, gamma, with added noise.
        """
        D = MMDataTransform.compute_fourier_kernel(ngrid) #Fourier kernel #TODO: Refactor back to ng in future - currently ng =
        sigma = 0.37 / np.sqrt(((theta*60/ngrid)**2)*ngal)
        gamma = MMDataTransform.forward_model(kappa, D) + sigma*(np.random.randn(ngrid,ngrid) + 1j * np.random.randn(ngrid,ngrid))
        return gamma

    def gamma_gen(self, kappa) -> np.ndarray:
        """Apply the forward model with the correct set of parameters.

        This function takes the input, kappa, applies the forward relationship to generate the corresponding value of the shear field,
        then adds Gaussian noise, to simulate a real observation.

        Args:
            kappa (np.ndarray): Convergence map/ground truth.
        
        Returns:
            gamma (np.ndarray): A synthetic representation of the shear field, gamma, with added noise.
        """
        return MMDataTransform.noise_maker(self.theta, self.ng, self.ngal, kappa)


    def __call__(self, kappa) -> Tuple[float, float, float, float]:
        """ Transforms the data.

        Note: gt = ground truth. The ground truth is the original kappa simulation from kappaTNG.
        Gamma represents the observation.

        Args:
            kappa (np.ndarray): Complex-valued array.

        Returns:
            (tuple) tuple containing:   
                normalized_gamma (float): Normalised measurement/gamma.
                normalized_gt (float): Normalised ground truth/kappa.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
        
        """
        # Generate observation on the fly.
        gamma = self.gamma_gen(kappa)

        # Format input gt data.
        pt_kappa = transforms.to_tensor(kappa) # Shape (H, W, 2)
        pt_kappa = pt_kappa.permute(2, 0, 1)  # Shape (2, H, W)
        # Format observation data.
        pt_gamma = transforms.to_tensor(gamma) # Shape (H, W, 2)
        pt_gamma = pt_gamma.permute(2, 0, 1)  # Shape (2, H, W)

        # Normalization step.
        normalized_gamma, mean, std = transforms.normalize_instance(pt_gamma)
        normalized_gt = transforms.normalize(pt_kappa, mean, std)

        # Return normalized measurements, normalized gt, mean, and std.
        # To unnormalize batch of images:
        # unnorm_tensor = normalized_tensor * std[:, :, None, None] + mean[:, :, None, None]
        return normalized_gamma.float(), normalized_gt.float(), mean, std




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
        #Assign train/val datasets for use in dataloaders

        train_data = MassMappingDataset_Train(
            data_dir=pathlib.Path(self.args.data_path) / 'kappa_train',
            transform=MMDataTransform(self.args, test=False)
        )

        dev_data = MassMappingDataset_Val(
            data_dir=pathlib.Path(self.args.data_path) / 'kappa_val',
            transform=MMDataTransform(self.args, test=True)
        )    

        test_data = MassMappingDataset_Test(
            data_dir=pathlib.Path(self.args.data_path) / 'kappa_test',
            transform=MMDataTransform(self.args, test=True)
        )

        self.train, self.validate, self.test = train_data, dev_data, test_data


    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validate,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            drop_last=False
        )
