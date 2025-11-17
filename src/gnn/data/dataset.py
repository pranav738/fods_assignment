"""
PyTorch Dataset for spatio-temporal traffic forecasting
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional


class TrafficGraphDataset(Dataset):
    """
    PyTorch Dataset for spatio-temporal graph data

    Handles batching and conversion to PyTorch tensors for GNN models.
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 adjacency: np.ndarray,
                 transform: Optional[callable] = None):
        """
        Initialize dataset

        Args:
            X: Input sequences (num_samples, window_size, num_nodes, num_features)
            y: Target sequences (num_samples, horizon, num_nodes, num_outputs)
            adjacency: Adjacency matrix (num_nodes, num_nodes)
            transform: Optional transform to apply to samples
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.adjacency = torch.FloatTensor(adjacency)
        self.transform = transform

        self.num_samples = X.shape[0]
        self.window_size = X.shape[1]
        self.num_nodes = X.shape[2]
        self.num_features = X.shape[3]
        self.horizon = y.shape[1]

    def __len__(self) -> int:
        """Return number of samples"""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample

        Args:
            idx: Sample index

        Returns:
            Tuple of (x, y, adj) where:
                x: Input sequence (window_size, num_nodes, num_features)
                y: Target sequence (horizon, num_nodes, num_outputs)
                adj: Adjacency matrix (num_nodes, num_nodes)
        """
        x = self.X[idx]
        y = self.y[idx]
        adj = self.adjacency

        if self.transform:
            x, y = self.transform(x, y)

        return x, y, adj

    def get_full_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get entire dataset as a single batch

        Returns:
            Tuple of (X, y, adjacency)
        """
        return self.X, self.y, self.adjacency


class SpatialCrossValidationDataset:
    """
    Dataset for spatial cross-validation

    Implements spatial holdout strategy where some nodes are held out
    during training to test spatial generalization.
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 adjacency: np.ndarray,
                 holdout_nodes: list,
                 all_nodes: list):
        """
        Initialize spatial CV dataset

        Args:
            X: Input sequences (num_samples, window_size, num_nodes, num_features)
            y: Target sequences (num_samples, horizon, num_nodes, num_outputs)
            adjacency: Adjacency matrix (num_nodes, num_nodes)
            holdout_nodes: List of node indices to hold out
            all_nodes: List of all node indices
        """
        self.X_full = X
        self.y_full = y
        self.adjacency_full = adjacency
        self.holdout_nodes = holdout_nodes
        self.train_nodes = [n for n in all_nodes if n not in holdout_nodes]

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training data (excluding holdout nodes)

        Returns:
            Tuple of (X_train, y_train, adjacency_train)
        """
        X_train = self.X_full[:, :, self.train_nodes, :]
        y_train = self.y_full[:, :, self.train_nodes, :]
        adj_train = self.adjacency_full[np.ix_(self.train_nodes, self.train_nodes)]

        return X_train, y_train, adj_train

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get test data (holdout nodes only)

        Returns:
            Tuple of (X_test, y_test)
        """
        X_test = self.X_full[:, :, self.holdout_nodes, :]
        y_test = self.y_full[:, :, self.holdout_nodes, :]

        return X_test, y_test


def collate_fn(batch):
    """
    Custom collate function for DataLoader

    Args:
        batch: List of samples from Dataset

    Returns:
        Batched tensors
    """
    x_batch = torch.stack([item[0] for item in batch])
    y_batch = torch.stack([item[1] for item in batch])
    adj = batch[0][2]  # Adjacency is same for all samples

    return x_batch, y_batch, adj


def load_metr_la_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load METR-LA dataset (for benchmarking and transfer learning)

    Args:
        data_dir: Directory containing METR-LA data files

    Returns:
        Tuple of (X, y, adjacency)
    """
    import os

    # Load adjacency matrix
    adj_file = os.path.join(data_dir, 'adj_mx.pkl')
    if os.path.exists(adj_file):
        import pickle
        with open(adj_file, 'rb') as f:
            _, _, adjacency = pickle.load(f, encoding='latin1')
    else:
        raise FileNotFoundError(f"Adjacency matrix not found: {adj_file}")

    # Load data
    data_file = os.path.join(data_dir, 'metr-la.h5')
    if os.path.exists(data_file):
        import h5py
        with h5py.File(data_file, 'r') as f:
            X = f['X'][:]
            y = f['y'][:]
    else:
        raise FileNotFoundError(f"Data file not found: {data_file}")

    return X, y, adjacency


def load_pems_bay_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PeMS-BAY dataset (for benchmarking and transfer learning)

    Args:
        data_dir: Directory containing PeMS-BAY data files

    Returns:
        Tuple of (X, y, adjacency)
    """
    import os

    # Load adjacency matrix
    adj_file = os.path.join(data_dir, 'adj_mx_bay.pkl')
    if os.path.exists(adj_file):
        import pickle
        with open(adj_file, 'rb') as f:
            _, _, adjacency = pickle.load(f, encoding='latin1')
    else:
        raise FileNotFoundError(f"Adjacency matrix not found: {adj_file}")

    # Load data
    data_file = os.path.join(data_dir, 'pems-bay.h5')
    if os.path.exists(data_file):
        import h5py
        with h5py.File(data_file, 'r') as f:
            X = f['X'][:]
            y = f['y'][:]
    else:
        raise FileNotFoundError(f"Data file not found: {data_file}")

    return X, y, adjacency
