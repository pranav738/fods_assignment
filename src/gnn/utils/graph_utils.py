"""
Graph utility functions for spatio-temporal traffic forecasting

Implements adjacency matrix normalization, graph feature computation,
and other graph-related operations.
"""

import numpy as np
import scipy.sparse as sp
import torch
from typing import Tuple, Optional, Union
import networkx as nx


def normalize_adjacency(adj_mx: Union[np.ndarray, sp.spmatrix],
                       method: str = 'symmetric') -> Union[np.ndarray, sp.spmatrix]:
    """
    Normalize adjacency matrix for graph convolution

    Args:
        adj_mx: Adjacency matrix (N x N)
        method: Normalization method ('symmetric', 'random_walk', or 'none')

    Returns:
        Normalized adjacency matrix

    Methods:
        - symmetric: D^(-1/2) @ A @ D^(-1/2) (used in GCN)
        - random_walk: D^(-1) @ A (used in some diffusion models)
        - none: Return original matrix
    """
    if method == 'none':
        return adj_mx

    # Convert to sparse matrix if needed
    if isinstance(adj_mx, np.ndarray):
        adj_mx = sp.csr_matrix(adj_mx)

    # Add self-loops (I + A)
    adj_mx = adj_mx + sp.eye(adj_mx.shape[0])

    # Compute degree matrix
    rowsum = np.array(adj_mx.sum(1))

    if method == 'symmetric':
        # D^(-1/2)
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # D^(-1/2) @ A @ D^(-1/2)
        normalized_adj = d_mat_inv_sqrt @ adj_mx @ d_mat_inv_sqrt

    elif method == 'random_walk':
        # D^(-1)
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        # D^(-1) @ A
        normalized_adj = d_mat_inv @ adj_mx

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized_adj


def compute_chebyshev_polynomials(adj_mx: np.ndarray,
                                  k: int = 3,
                                  lambda_max: Optional[float] = None) -> list:
    """
    Compute Chebyshev polynomials of the graph Laplacian

    Used in some spectral graph convolution methods (ChebNet)

    Args:
        adj_mx: Adjacency matrix
        k: Order of Chebyshev polynomial
        lambda_max: Maximum eigenvalue (computed if not provided)

    Returns:
        List of Chebyshev polynomial matrices
    """
    # Compute normalized Laplacian
    adj_normalized = normalize_adjacency(adj_mx, method='symmetric')

    if sp.issparse(adj_normalized):
        adj_normalized = adj_normalized.toarray()

    L = np.eye(adj_mx.shape[0]) - adj_normalized

    # Compute maximum eigenvalue if not provided
    if lambda_max is None:
        lambda_max = sp.linalg.eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]

    # Rescale Laplacian: L_rescaled = 2/lambda_max * L - I
    L = 2 * L / lambda_max - np.eye(adj_mx.shape[0])

    # Compute Chebyshev polynomials: T_k(L)
    cheb_polynomials = [np.eye(adj_mx.shape[0]), L]

    for i in range(2, k):
        cheb_polynomials.append(2 * L @ cheb_polynomials[-1] - cheb_polynomials[-2])

    return cheb_polynomials


def compute_diffusion_matrix(adj_mx: np.ndarray,
                             steps: int = 2,
                             method: str = 'forward') -> list:
    """
    Compute random walk diffusion matrices

    Used in DCRNN for modeling traffic diffusion on the road network

    Args:
        adj_mx: Adjacency matrix
        steps: Number of diffusion steps (K in the paper)
        method: 'forward' or 'backward' diffusion

    Returns:
        List of diffusion matrices [P^0, P^1, ..., P^K]
    """
    num_nodes = adj_mx.shape[0]

    # Normalize adjacency for random walk
    adj_normalized = normalize_adjacency(adj_mx, method='random_walk')

    if sp.issparse(adj_normalized):
        adj_normalized = adj_normalized.toarray()

    # Compute diffusion matrices
    if method == 'forward':
        # Forward diffusion: P_f = D^(-1) @ A
        diffusion_matrix = adj_normalized
    elif method == 'backward':
        # Backward diffusion: P_b = A @ D^(-1) = (D^(-1) @ A^T)
        diffusion_matrix = adj_normalized.T
    else:
        raise ValueError(f"Unknown diffusion method: {method}")

    # Compute powers of diffusion matrix
    diffusion_matrices = [np.eye(num_nodes)]
    current_matrix = np.eye(num_nodes)

    for _ in range(steps):
        current_matrix = current_matrix @ diffusion_matrix
        diffusion_matrices.append(current_matrix.copy())

    return diffusion_matrices


def compute_graph_features(G: nx.Graph) -> dict:
    """
    Compute graph-theoretic features for each node

    These can be used as static node features in the GNN models.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary of node features
    """
    features = {}

    # Degree centrality
    features['degree_centrality'] = nx.degree_centrality(G)

    # Betweenness centrality (critical junctions)
    features['betweenness_centrality'] = nx.betweenness_centrality(G)

    # Closeness centrality
    features['closeness_centrality'] = nx.closeness_centrality(G)

    # PageRank (importance in the network)
    features['pagerank'] = nx.pagerank(G)

    # Clustering coefficient
    features['clustering'] = nx.clustering(G)

    return features


def spatial_distance_weight(distance: float,
                           sigma: float = 10.0,
                           epsilon: float = 0.5) -> float:
    """
    Compute spatial weight based on distance

    Uses Gaussian kernel: w = exp(-(distance^2) / (2 * sigma^2))

    Args:
        distance: Distance between nodes (in km)
        sigma: Bandwidth parameter
        epsilon: Threshold (edges with weight < epsilon are set to 0)

    Returns:
        Spatial weight
    """
    weight = np.exp(-np.square(distance) / (2 * np.square(sigma)))
    return weight if weight >= epsilon else 0.0


def build_spatial_adjacency(distances: np.ndarray,
                           sigma: float = 10.0,
                           epsilon: float = 0.5) -> np.ndarray:
    """
    Build spatial adjacency matrix from distance matrix

    Args:
        distances: Distance matrix (N x N)
        sigma: Gaussian kernel bandwidth
        epsilon: Weight threshold

    Returns:
        Spatial adjacency matrix
    """
    num_nodes = distances.shape[0]
    adj_mx = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                adj_mx[i, j] = spatial_distance_weight(distances[i, j], sigma, epsilon)

    return adj_mx


def transition_matrix(adj_mx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute forward and backward random walk transition matrices

    Used in DCRNN for bidirectional diffusion

    Args:
        adj_mx: Adjacency matrix

    Returns:
        Tuple of (forward_transition, backward_transition)
    """
    # Add self-loops
    adj_mx = adj_mx + np.eye(adj_mx.shape[0])

    # Compute row-normalized adjacency (forward)
    d = np.sum(adj_mx, axis=1)
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.diag(d_inv)
    forward = d_mat_inv @ adj_mx

    # Compute column-normalized adjacency (backward)
    backward = adj_mx @ d_mat_inv

    return forward, backward


def sparse_to_tuple(sparse_mx: sp.spmatrix) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """
    Convert sparse matrix to tuple representation for PyTorch

    Args:
        sparse_mx: Scipy sparse matrix

    Returns:
        Tuple of (indices, values, shape)
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    indices = np.vstack((sparse_mx.row, sparse_mx.col))
    values = sparse_mx.data
    shape = sparse_mx.shape

    return indices, values, shape


def adj_to_torch_sparse(adj_mx: Union[np.ndarray, sp.spmatrix],
                        device: torch.device = torch.device('cpu')) -> torch.sparse.FloatTensor:
    """
    Convert adjacency matrix to PyTorch sparse tensor

    Args:
        adj_mx: Adjacency matrix (numpy or scipy sparse)
        device: PyTorch device

    Returns:
        PyTorch sparse tensor
    """
    if isinstance(adj_mx, np.ndarray):
        adj_mx = sp.coo_matrix(adj_mx)

    indices, values, shape = sparse_to_tuple(adj_mx)

    indices = torch.LongTensor(indices).to(device)
    values = torch.FloatTensor(values).to(device)
    shape = torch.Size(shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def calculate_laplacian(adj_mx: np.ndarray,
                       normalize: bool = True) -> np.ndarray:
    """
    Calculate graph Laplacian matrix

    L = D - A (unnormalized)
    L = I - D^(-1/2) @ A @ D^(-1/2) (normalized)

    Args:
        adj_mx: Adjacency matrix
        normalize: Whether to compute normalized Laplacian

    Returns:
        Laplacian matrix
    """
    if normalize:
        # Normalized Laplacian
        adj_normalized = normalize_adjacency(adj_mx, method='symmetric')
        if sp.issparse(adj_normalized):
            adj_normalized = adj_normalized.toarray()
        laplacian = np.eye(adj_mx.shape[0]) - adj_normalized
    else:
        # Unnormalized Laplacian
        d = np.sum(adj_mx, axis=1)
        laplacian = np.diag(d) - adj_mx

    return laplacian


def load_adjacency_matrix(file_path: str,
                         dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Load adjacency matrix from file

    Supports .npy, .npz, .csv formats

    Args:
        file_path: Path to adjacency matrix file
        dtype: Data type for the matrix

    Returns:
        Adjacency matrix
    """
    if file_path.endswith('.npy'):
        adj_mx = np.load(file_path)
    elif file_path.endswith('.npz'):
        data = np.load(file_path)
        adj_mx = data['adjacency']
    elif file_path.endswith('.csv'):
        adj_mx = np.loadtxt(file_path, delimiter=',')
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    return adj_mx.astype(dtype)


def save_adjacency_matrix(adj_mx: np.ndarray,
                         file_path: str,
                         compressed: bool = True):
    """
    Save adjacency matrix to file

    Args:
        adj_mx: Adjacency matrix
        file_path: Output file path
        compressed: Whether to save as compressed .npz
    """
    if compressed or file_path.endswith('.npz'):
        np.savez_compressed(file_path, adjacency=adj_mx)
    else:
        np.save(file_path, adj_mx)
