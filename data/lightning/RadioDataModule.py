import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional
from data.datasets.Radio_data import RadioDataset_Test, RadioDataset_Train, RadioDataset_Val
from utils.mri import transforms
from typing import Tuple
import pathlib



class RadioDataTransform:
    def __init__(self, args, test=False, ISNR=30):
        self.args = args
        self.test = test
        self.ISNR = ISNR

    def __call__(self, data) -> Tuple[float, float, float, float]:
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
        
        x, y, uv = data



        # Format input gt data.
        pt_x = transforms.to_tensor(x) # Shape (H, W, 2)
        pt_x = pt_x.permute(2, 0, 1)  # Shape (2, H, W)
        # Format observation data.
        pt_y = transforms.to_tensor(y) # Shape (H, W, 2)
        pt_y = pt_y.permute(2, 0, 1)  # Shape (2, H, W)

        # Normalization step.
        normalized_y, mean, std = transforms.normalize_instance(pt_y)
        normalized_x = transforms.normalize(pt_x, mean, std)

        # Return normalized measurements, normalized gt, mean, and std.
        # To unnormalize batch of images:
        # unnorm_tensor = normalized_tensor * std[:, :, None, None] + mean[:, :, None, None]
        return normalized_y.float(), normalized_x.float(), mean, std




class RadioDataModule(pl.LightningDataModule):
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

        train_data = RadioDataset_Train(
            data_dir=pathlib.Path(self.args.data_path) / 'train',
            transform=RadioDataTransform(self.args, test=False)
        )

        dev_data = RadioDataset_Val(
            data_dir=pathlib.Path(self.args.data_path) / 'val',
            transform=RadioDataTransform(self.args, test=True)
        )    

        test_data = RadioDataset_Test(
            data_dir=pathlib.Path(self.args.data_path) / 'test',
            transform=RadioDataTransform(self.args, test=True)
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
