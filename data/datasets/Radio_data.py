# Add data set information and configuration here.
import numpy as np
import torch
import pathlib


class RadioDataset_Test(torch.utils.data.Dataset):
    """Loads the test data."""
    def __init__(self, data_dir, transform):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object (class) that pre-processes the raw data into
                appropriate form for it to be fed into the model.
        """
        self.transform = transform
        
        # Collects the paths of all files.
        # Test/x.npy, Test/y.npy, Test/uv.npy
        self.x = np.load(data_dir.joinpath("x.npy")).astype(np.complex128)
        self.y = np.load(data_dir.joinpath("y.npy")).astype(np.complex128)
        self.uv = np.load(data_dir.joinpath("uv.npy")).real.astype(np.float64)
        self.uv = (self.uv - self.uv.min())/(self.uv.max() - self.uv.min()) # normalize range of uv values to (0,1)       
        

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
    def __init__(self, data_dir, transform):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object (class) that pre-processes the raw data into
                appropriate form for it to be fed into the model.
        """
        self.transform = transform
        
        # Collects the paths of all files.
        # Val/x.npy, Val/y.npy, Val/uv.npy
        self.x = np.load(data_dir.joinpath("x.npy")).astype(np.complex128)
        self.y = np.load(data_dir.joinpath("y.npy")).astype(np.complex128)
        self.uv = np.load(data_dir.joinpath("uv.npy")).real.astype(np.float64)
        self.uv = (self.uv - self.uv.min())/(self.uv.max() - self.uv.min()) # normalize range of uv values to (0,1)       

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
    def __init__(self, data_dir, transform):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object (class) that pre-processes the raw data into
                appropriate form for it to be fed into the model.
        """
        self.transform = transform
        
        # Collects the paths of all files.
        # Train/x.npy, Train/y.npy, Train/uv.npy
        self.x = np.load(data_dir.joinpath("x.npy")).astype(np.complex128)
        self.y = np.load(data_dir.joinpath("y.npy")).astype(np.complex128)
        self.uv = np.load(data_dir.joinpath("uv.npy")).real.astype(np.float64)
        self.uv = (self.uv - self.uv.min())/(self.uv.max() - self.uv.min()) # normalize range of uv values to (0,1)       
        

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.x)
    
    def __getitem__(self,i):
        """Loads and returns a sample from the dataset at a given index."""
        data = (self.x[i], self.y[i], self.uv[i])
        # Cast input data from float64 to complex128 as we require complex dtype.
        # Tranform data and generate observations.
        return self.transform(data)