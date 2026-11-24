# Add data set information and configuration here.
import numpy as np
import torch
import pathlib


class MassMappingDataset(torch.utils.data.Dataset):
    """Loads one split (train, val or test) of the convergence map dataset."""

    def __init__(self, data_dir, transform):
        """
        Args:
            data_dir (path): The path to the split directory (containing kappa_*.npy).
            transform (callable): A callable object (class) that pre-processes the raw data into
                appropriate form for it to be fed into the model.
        """
        self.transform = transform

        # Collects the paths of all maps in a fixed (sorted) order. The generator
        # already writes them in a seeded random order; per-epoch shuffling of the
        # training set is done by the DataLoader.
        self.examples = sorted(pathlib.Path(data_dir).glob("kappa_*.npy"))

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.examples)

    def __getitem__(self, i):
        """Loads and returns a sample from the dataset at a given index."""

        data = np.load(self.examples[i], allow_pickle=True).astype(np.float64)
        # Tranform data and generate observations.
        return self.transform(data)
