# Add data set information and configuration here.
import numpy as np
import torch
import pathlib


class RadioDataset_Test(torch.utils.data.Dataset):
    """Loads the test data."""
    def __init__(self, data_dir, transform, norm='micro'):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object (class) that pre-processes the raw data into
                appropriate form for it to be fed into the model.
            norm (str): either 'none' (no normalisation), 'micro' (per sample normalisation), 'macro' (normalisation across all samples)
        """
        self.transform = transform
        
        # Collects the paths of all files.
        # Test/x.npy, Test/y.npy, Test/uv.npy
        self.x = np.load(data_dir.joinpath("x.npy")).astype(np.float64)
        self.y = np.load(data_dir.joinpath("y.npy")).astype(np.float64)
        self.uv = np.load(data_dir.joinpath("uv.npy")).real.astype(np.float64)

        if norm == 'none':
            self.transform.mean_x, self.transform.std_x = 0, 1
            self.transform.mean_y, self.transform.std_y = 0, 1
            self.transform.mean_uv, self.transform.std_uv = 0, 1
        elif norm == 'micro':
            # if micro we do the normalisation in the transform
            pass
        elif norm == 'macro':
            # load means and stds from train set
            self.transform.mean_x  = np.load(data_dir.parent.joinpath("train/mean_x.npy"))
            self.transform.std_x   = np.load(data_dir.parent.joinpath("train/std_x.npy"))
            self.transform.mean_y  = np.load(data_dir.parent.joinpath("train/mean_y.npy"))
            self.transform.std_y   = np.load(data_dir.parent.joinpath("train/std_y.npy"))
            self.transform.mean_uv = np.load(data_dir.parent.joinpath("train/mean_uv.npy"))
            self.transform.std_uv  = np.load(data_dir.parent.joinpath("train/std_uv.npy"))
            
#             self.transform.mean_x, self.transform.std_x = self.x.mean(), self.x.std()
#             self.transform.mean_y, self.transform.std_y = self.y.mean(), self.y.std()
#             self.transform.mean_uv, self.transform.std_uv = self.uv.mean(), self.uv.std()

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.x)
    
    def __getitem__(self,i):
        """Loads and returns a sample from the dataset at a given index."""
        data = (self.x[i], self.y[i], self.uv[i])
        # Cast input data from float64 to complex128 as we require complex dtype.
        # Tranform data and generate observations.
        return self.transform(data)


class RadioDataset_Val(torch.utils.data.Dataset):
    """Loads the test data."""
    def __init__(self, data_dir, transform, norm='micro'):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object (class) that pre-processes the raw data into
                appropriate form for it to be fed into the model.
            norm (str): either 'none' (no normalisation), 'micro' (per sample normalisation), 'macro' (normalisation across all samples)
        """
        self.transform = transform
        
        # Collects the paths of all files.
        # Val/x.npy, Val/y.npy, Val/uv.npy
        self.x = np.load(data_dir.joinpath("x.npy")).astype(np.float64)
        self.y = np.load(data_dir.joinpath("y.npy")).astype(np.float64)
        self.uv = np.load(data_dir.joinpath("uv.npy")).real.astype(np.float64)

        if norm == 'none':
            self.transform.mean_x, self.transform.std_x = 0, 1
            self.transform.mean_y, self.transform.std_y = 0, 1
            self.transform.mean_uv, self.transform.std_uv = 0, 1
        elif norm == 'micro':
            # if micro we do the normalisation in the transform
            pass
        elif norm == 'macro':
            # load means and stds from train set
            self.transform.mean_x  = np.load(data_dir.parent.joinpath("train/mean_x.npy"))
            self.transform.std_x   = np.load(data_dir.parent.joinpath("train/std_x.npy"))
            self.transform.mean_y  = np.load(data_dir.parent.joinpath("train/mean_y.npy"))
            self.transform.std_y   = np.load(data_dir.parent.joinpath("train/std_y.npy"))
            self.transform.mean_uv = np.load(data_dir.parent.joinpath("train/mean_uv.npy"))
            self.transform.std_uv  = np.load(data_dir.parent.joinpath("train/std_uv.npy"))
            
#             self.transform.mean_x, self.transform.std_x = self.x.mean(), self.x.std()
#             self.transform.mean_y, self.transform.std_y = self.y.mean(), self.y.std()
#             self.transform.mean_uv, self.transform.std_uv = self.uv.mean(), self.uv.std()

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.x)
    
    def __getitem__(self,i):
        """Loads and returns a sample from the dataset at a given index."""
        data = (self.x[i], self.y[i], self.uv[i])
        # Cast input data from float64 to complex128 as we require complex dtype.
        # Tranform data and generate observations.
        return self.transform(data)
    
class RadioDataset_Train(torch.utils.data.Dataset):
    """Loads the test data."""
    def __init__(self, data_dir, transform, norm='micro'):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object (class) that pre-processes the raw data into
                appropriate form for it to be fed into the model.
            norm (str): either 'none' (no normalisation), 'micro' (per sample normalisation), 'macro' (normalisation across all samples)
        """
        self.transform = transform
        
        # Collects the paths of all files.
        # Train/x.npy, Train/y.npy, Train/uv.npy
        self.x = np.load(data_dir.joinpath("x.npy")).astype(np.float64)
        self.y = np.load(data_dir.joinpath("y.npy")).astype(np.float64)
        self.uv = np.load(data_dir.joinpath("uv.npy")).real.astype(np.float64)

        if norm == 'none':
            self.transform.mean_x, self.transform.std_x = 0, 1
            self.transform.mean_y, self.transform.std_y = 0, 1
            self.transform.mean_uv, self.transform.std_uv = 0, 1
        elif norm == 'micro':
            # if micro we do the normalisation in the transform
            pass
        elif norm == 'macro':
            self.transform.mean_x, self.transform.std_x = self.x.mean(), np.mean(self.x.std(axis=(1,2)))
            self.transform.mean_y, self.transform.std_y = self.y.mean(), np.mean(self.y.std(axis=(1,2)))
            self.transform.mean_uv, self.transform.std_uv = self.uv.mean(),  np.mean(self.uv.std(axis=(1,2)))
            
            np.save(data_dir.joinpath("mean_x.npy"), self.transform.mean_x)
            np.save(data_dir.joinpath("std_x.npy"), self.transform.std_x)
            np.save(data_dir.joinpath("mean_y.npy"), self.transform.mean_y)
            np.save(data_dir.joinpath("std_y.npy"), self.transform.std_y)
            np.save(data_dir.joinpath("mean_uv.npy"), self.transform.mean_uv)
            np.save(data_dir.joinpath("std_uv.npy"), self.transform.std_uv)
            
            
            

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.x)
    
    def __getitem__(self,i):
        """Loads and returns a sample from the dataset at a given index."""
        data = (self.x[i], self.y[i], self.uv[i])
        # Cast input data from float64 to complex128 as we require complex dtype.
        # Tranform data and generate observations.
        return self.transform(data)