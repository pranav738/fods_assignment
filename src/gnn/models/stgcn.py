"""
Spatio-Temporal Graph Convolutional Network (STGCN)

Reference:
Yu, B., Yin, H., & Zhu, Z. (2018). Spatio-temporal graph convolutional networks:
A deep learning framework for traffic forecasting. IJCAI 2018.

Key Components:
1. Graph Convolution (GCN) for spatial dependencies
2. 1D Gated Convolution for temporal dependencies
3. Stacking of ST-Conv blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer

    Implements: H' = σ(D^(-1/2) @ A @ D^(-1/2) @ H @ W)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize graph convolution layer

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to add bias
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features (batch_size, num_nodes, in_features)
            adj: Normalized adjacency matrix (num_nodes, num_nodes)

        Returns:
            Output features (batch_size, num_nodes, out_features)
        """
        # x: (B, N, F_in)
        # adj: (N, N)
        # weight: (F_in, F_out)

        # Linear transformation: x @ W
        support = torch.matmul(x, self.weight)  # (B, N, F_out)

        # Graph convolution: adj @ support
        output = torch.matmul(adj, support)  # (B, N, F_out)

        if self.bias is not None:
            output = output + self.bias

        return output


class TemporalConvLayer(nn.Module):
    """
    Temporal Convolutional Layer

    Uses 1D gated convolution: GLU(X) = (X * W + b) ⊙ σ(X * V + c)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3):
        """
        Initialize temporal convolution layer

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
        """
        super(TemporalConvLayer, self).__init__()

        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        # Gated Linear Unit: requires 2x output channels
        self.conv = nn.Conv2d(
            in_channels,
            2 * out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, self.padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, channels, num_nodes, time_steps)

        Returns:
            Output tensor (batch_size, out_channels, num_nodes, time_steps)
        """
        # Apply convolution
        out = self.conv(x)  # (B, 2*C_out, N, T)

        # Split into two parts for gating
        P, Q = torch.chunk(out, 2, dim=1)  # Each: (B, C_out, N, T)

        # Gated Linear Unit: P ⊙ sigmoid(Q)
        out = P * torch.sigmoid(Q)

        return out


class STConvBlock(nn.Module):
    """
    Spatio-Temporal Convolutional Block

    Structure: TCN -> GCN -> TCN
    """

    def __init__(self,
                 num_nodes: int,
                 in_channels: int,
                 spatial_channels: int,
                 out_channels: int,
                 num_temporal_layers: int = 3):
        """
        Initialize ST-Conv block

        Args:
            num_nodes: Number of nodes in the graph
            in_channels: Input channels
            spatial_channels: Intermediate spatial channels
            out_channels: Output channels
            num_temporal_layers: Number of temporal layers
        """
        super(STConvBlock, self).__init__()

        self.temporal1 = TemporalConvLayer(in_channels, out_channels)
        self.graph_conv = GraphConvolution(out_channels, spatial_channels)
        self.temporal2 = TemporalConvLayer(spatial_channels, out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, in_channels, num_nodes, time_steps)
            adj: Adjacency matrix (num_nodes, num_nodes)

        Returns:
            Output tensor (batch_size, out_channels, num_nodes, time_steps)
        """
        # Temporal convolution 1
        t = self.temporal1(x)  # (B, C_out, N, T)

        # Transpose for graph convolution: (B, C_out, N, T) -> (B, T, N, C_out)
        t2 = t.permute(0, 3, 2, 1)  # (B, T, N, C_out)

        # Apply graph convolution to each time step
        t3 = torch.stack([
            self.graph_conv(t2[:, i, :, :], adj)
            for i in range(t2.shape[1])
        ], dim=1)  # (B, T, N, C_spatial)

        # Transpose back: (B, T, N, C_spatial) -> (B, C_spatial, N, T)
        t4 = t3.permute(0, 3, 2, 1)  # (B, C_spatial, N, T)

        # Temporal convolution 2
        t5 = self.temporal2(t4)  # (B, C_out, N, T)

        # Batch normalization
        out = self.batch_norm(t5.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        return out


class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network

    End-to-end model for traffic forecasting.
    """

    def __init__(self,
                 num_nodes: int,
                 in_channels: int,
                 spatial_channels: int = 16,
                 temporal_channels: int = 64,
                 num_blocks: int = 2,
                 horizon: int = 12,
                 dropout: float = 0.3):
        """
        Initialize STGCN model

        Args:
            num_nodes: Number of nodes in the graph
            in_channels: Number of input features per node
            spatial_channels: Channels for graph convolution
            temporal_channels: Channels for temporal convolution
            num_blocks: Number of ST-Conv blocks
            horizon: Prediction horizon (number of future time steps)
            dropout: Dropout rate
        """
        super(STGCN, self).__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.spatial_channels = spatial_channels
        self.temporal_channels = temporal_channels
        self.num_blocks = num_blocks
        self.horizon = horizon

        # ST-Conv blocks
        self.st_blocks = nn.ModuleList()

        # First block
        self.st_blocks.append(
            STConvBlock(
                num_nodes,
                in_channels,
                spatial_channels,
                temporal_channels
            )
        )

        # Subsequent blocks
        for _ in range(1, num_blocks):
            self.st_blocks.append(
                STConvBlock(
                    num_nodes,
                    temporal_channels,
                    spatial_channels,
                    temporal_channels
                )
            )

        # Output layer
        self.final_conv = nn.Conv2d(
            temporal_channels,
            horizon,
            kernel_size=(1, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, window_size, num_nodes, in_channels)
            adj: Adjacency matrix (num_nodes, num_nodes)

        Returns:
            Predictions (batch_size, horizon, num_nodes, 1)
        """
        # Transpose to (B, C, N, T) format
        x = x.permute(0, 3, 2, 1)  # (B, in_channels, N, window_size)

        # Apply ST-Conv blocks
        for block in self.st_blocks:
            x = block(x, adj)
            x = self.dropout(x)

        # Final convolution: (B, C, N, T) -> (B, horizon, N, 1)
        x = self.final_conv(x)  # (B, horizon, N, 1)

        # Transpose to (B, horizon, N, 1)
        x = x.permute(0, 1, 2, 3)

        return x


class STGCNChebConv(nn.Module):
    """
    STGCN variant using Chebyshev polynomial graph convolution

    This variant uses spectral graph convolution with Chebyshev polynomials
    instead of simple graph convolution.
    """

    def __init__(self,
                 num_nodes: int,
                 in_channels: int,
                 spatial_channels: int = 16,
                 temporal_channels: int = 64,
                 num_blocks: int = 2,
                 horizon: int = 12,
                 K: int = 3,
                 dropout: float = 0.3):
        """
        Initialize STGCN with Chebyshev convolution

        Args:
            num_nodes: Number of nodes
            in_channels: Input feature channels
            spatial_channels: Spatial convolution channels
            temporal_channels: Temporal convolution channels
            num_blocks: Number of ST-Conv blocks
            horizon: Prediction horizon
            K: Order of Chebyshev polynomial
            dropout: Dropout rate
        """
        super(STGCNChebConv, self).__init__()

        # Implementation would use torch_geometric.nn.ChebConv
        # Placeholder for now
        self.model = STGCN(
            num_nodes, in_channels, spatial_channels,
            temporal_channels, num_blocks, horizon, dropout
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x, adj)


def create_stgcn_model(config: dict) -> STGCN:
    """
    Factory function to create STGCN model from config

    Args:
        config: Configuration dictionary

    Returns:
        STGCN model
    """
    return STGCN(
        num_nodes=config.get('num_nodes'),
        in_channels=config.get('in_channels'),
        spatial_channels=config.get('spatial_channels', 16),
        temporal_channels=config.get('temporal_channels', 64),
        num_blocks=config.get('num_blocks', 2),
        horizon=config.get('horizon', 12),
        dropout=config.get('dropout', 0.3)
    )
