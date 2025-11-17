"""
Attention Based Spatial-Temporal Graph Convolutional Networks (ASTGCN)

Reference:
Guo, S., Lin, Y., Feng, N., Song, C., & Wan, H. (2019).
Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting.
AAAI 2019.

Key Features:
1. Spatial attention mechanism
2. Temporal attention mechanism
3. Multi-component fusion (recent, daily-periodic, weekly-periodic)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SpatialAttention(nn.Module):
    """
    Spatial Attention Mechanism

    Learns which nodes (locations) are most relevant for prediction
    """

    def __init__(self, num_nodes: int, num_features: int, num_timesteps: int):
        """
        Initialize spatial attention

        Args:
            num_nodes: Number of nodes
            num_features: Number of input features
            num_timesteps: Number of time steps
        """
        super(SpatialAttention, self).__init__()

        self.W1 = nn.Parameter(torch.FloatTensor(num_timesteps))
        self.W2 = nn.Parameter(torch.FloatTensor(num_features, num_timesteps))
        self.W3 = nn.Parameter(torch.FloatTensor(num_features))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_nodes, num_nodes))
        self.Vs = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.uniform_(self.W3)
        nn.init.uniform_(self.bs)
        nn.init.uniform_(self.Vs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention matrix

        Args:
            x: Input (batch_size, num_nodes, num_features, num_timesteps)

        Returns:
            Spatial attention matrix (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, num_features, num_timesteps = x.shape

        # Compute attention scores
        # lhs: (B, N, T) @ (T,) -> (B, N)
        lhs = torch.matmul(x, self.W1)

        # rhs: (B, N, F, T) @ (F, T) -> (B, N, T) @ (F,) via transpose
        rhs = torch.matmul(
            torch.matmul(x.permute(0, 1, 3, 2), self.W2.T).permute(0, 1, 3, 2),
            self.W3
        )

        # Product: (B, N, 1) @ (B, 1, N) -> (B, N, N)
        product = torch.matmul(
            lhs.unsqueeze(2),
            rhs.unsqueeze(1)
        )

        # Attention scores
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))

        # Normalize using softmax
        S = F.softmax(S, dim=-1)

        return S


class TemporalAttention(nn.Module):
    """
    Temporal Attention Mechanism

    Learns which time steps are most relevant for prediction
    """

    def __init__(self, num_nodes: int, num_features: int, num_timesteps: int):
        """
        Initialize temporal attention

        Args:
            num_nodes: Number of nodes
            num_features: Number of features
            num_timesteps: Number of time steps
        """
        super(TemporalAttention, self).__init__()

        self.U1 = nn.Parameter(torch.FloatTensor(num_nodes))
        self.U2 = nn.Parameter(torch.FloatTensor(num_features, num_nodes))
        self.U3 = nn.Parameter(torch.FloatTensor(num_features))
        self.be = nn.Parameter(torch.FloatTensor(1, num_timesteps, num_timesteps))
        self.Ve = nn.Parameter(torch.FloatTensor(num_timesteps, num_timesteps))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.uniform_(self.U1)
        nn.init.xavier_uniform_(self.U2)
        nn.init.uniform_(self.U3)
        nn.init.uniform_(self.be)
        nn.init.uniform_(self.Ve)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal attention matrix

        Args:
            x: Input (batch_size, num_nodes, num_features, num_timesteps)

        Returns:
            Temporal attention matrix (batch_size, num_timesteps, num_timesteps)
        """
        batch_size, num_nodes, num_features, num_timesteps = x.shape

        # Transpose: (B, N, F, T) -> (B, T, F, N)
        x_transposed = x.permute(0, 3, 2, 1)

        # Compute attention scores
        # lhs: (B, T, N) @ (N,) -> (B, T)
        lhs = torch.matmul(x_transposed.sum(dim=2), self.U1)

        # rhs: similar computation
        rhs = torch.matmul(
            torch.matmul(x_transposed, self.U2.T).sum(dim=2),
            self.U3
        )

        # Product: (B, T, 1) @ (B, 1, T) -> (B, T, T)
        product = torch.matmul(
            lhs.unsqueeze(2),
            rhs.unsqueeze(1)
        )

        # Attention scores
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))

        # Normalize using softmax
        E = F.softmax(E, dim=-1)

        return E


class ChebGraphConv(nn.Module):
    """
    Chebyshev Graph Convolution

    Uses Chebyshev polynomials for spectral graph convolution
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 3):
        """
        Initialize Chebyshev graph convolution

        Args:
            in_channels: Input channels
            out_channels: Output channels
            K: Order of Chebyshev polynomial
        """
        super(ChebGraphConv, self).__init__()

        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Weight for each Chebyshev polynomial
        self.weights = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input (batch_size, num_nodes, in_channels)
            laplacian: Scaled Laplacian matrix (num_nodes, num_nodes)

        Returns:
            Output (batch_size, num_nodes, out_channels)
        """
        batch_size, num_nodes, in_channels = x.shape

        # Compute Chebyshev polynomials
        # T_0 = I
        Tx_0 = x
        outputs = torch.matmul(Tx_0, self.weights[0])

        if self.K > 1:
            # T_1 = L
            Tx_1 = torch.matmul(laplacian, x)
            outputs = outputs + torch.matmul(Tx_1, self.weights[1])

        # T_k = 2 * L * T_{k-1} - T_{k-2}
        for k in range(2, self.K):
            Tx_2 = 2 * torch.matmul(laplacian, Tx_1) - Tx_0
            outputs = outputs + torch.matmul(Tx_2, self.weights[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        return outputs


class ASTGCNBlock(nn.Module):
    """
    ASTGCN Block

    Combines spatial attention, temporal attention, and graph convolution
    """

    def __init__(self,
                 num_nodes: int,
                 in_channels: int,
                 out_channels: int,
                 num_timesteps: int,
                 K: int = 3):
        """
        Initialize ASTGCN block

        Args:
            num_nodes: Number of nodes
            in_channels: Input channels
            out_channels: Output channels
            num_timesteps: Number of time steps
            K: Chebyshev polynomial order
        """
        super(ASTGCNBlock, self).__init__()

        self.temporal_attention = TemporalAttention(num_nodes, in_channels, num_timesteps)
        self.spatial_attention = SpatialAttention(num_nodes, in_channels, num_timesteps)
        self.cheb_conv = ChebGraphConv(in_channels, out_channels, K)
        self.time_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, num_timesteps)
        )
        self.residual_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1)
        )
        self.layer_norm = nn.LayerNorm([num_nodes, out_channels])

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input (batch_size, num_nodes, in_channels, num_timesteps)
            laplacian: Graph Laplacian (num_nodes, num_nodes)

        Returns:
            Output (batch_size, num_nodes, out_channels)
        """
        batch_size, num_nodes, in_channels, num_timesteps = x.shape

        # Temporal attention
        temporal_att = self.temporal_attention(x)  # (B, T, T)

        # Apply temporal attention: (B, T, T) @ (B, T, N, F) -> (B, T, N, F)
        x_TAt = torch.matmul(
            temporal_att,
            x.permute(0, 3, 1, 2)
        )  # (B, T, N, F)
        x_TAt = x_TAt.permute(0, 2, 3, 1)  # (B, N, F, T)

        # Spatial attention
        spatial_att = self.spatial_attention(x)  # (B, N, N)

        # Graph convolution with spatial attention
        # Process each time step
        spatial_gcn = []
        for t in range(num_timesteps):
            # Apply spatial attention
            x_t = x_TAt[:, :, :, t]  # (B, N, F)

            # Spatial-attention weighted adjacency
            # S_att @ X @ W
            x_SAtt = torch.matmul(spatial_att, x_t)  # (B, N, F)

            # Graph convolution
            gcn_out = self.cheb_conv(x_SAtt, laplacian)  # (B, N, out_channels)
            spatial_gcn.append(gcn_out)

        # Stack: (B, N, out_channels, T)
        spatial_gcn = torch.stack(spatial_gcn, dim=-1)

        # Temporal convolution
        # (B, in_channels, N, T) -> (B, out_channels, N, 1)
        time_conv_output = self.time_conv(x.permute(0, 2, 1, 3))
        time_conv_output = time_conv_output.squeeze(-1).permute(0, 2, 1)  # (B, N, out_channels)

        # Spatial-temporal convolution output
        # Average over time
        spatial_gcn_avg = spatial_gcn.mean(dim=-1)  # (B, N, out_channels)

        # Residual connection
        if in_channels != out_channels:
            x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
            x_residual = x_residual.squeeze(-1).permute(0, 2, 1)
        else:
            x_residual = x[:, :, :, -1]  # Take last time step

        # Combine
        output = spatial_gcn_avg + time_conv_output + x_residual

        # Layer normalization
        output = self.layer_norm(output)

        return F.relu(output)


class ASTGCN(nn.Module):
    """
    Attention Based Spatial-Temporal Graph Convolutional Network

    Multi-component fusion architecture
    """

    def __init__(self,
                 num_nodes: int,
                 in_channels: int,
                 nb_block: int = 2,
                 nb_chev_filter: int = 64,
                 nb_time_filter: int = 64,
                 time_strides: int = 1,
                 num_timesteps_input: int = 12,
                 num_timesteps_output: int = 12,
                 K: int = 3):
        """
        Initialize ASTGCN

        Args:
            num_nodes: Number of nodes
            in_channels: Input feature channels
            nb_block: Number of ASTGCN blocks
            nb_chev_filter: Number of Chebyshev filters
            nb_time_filter: Number of temporal filters
            time_strides: Temporal stride
            num_timesteps_input: Input sequence length
            num_timesteps_output: Output sequence length
            K: Chebyshev polynomial order
        """
        super(ASTGCN, self).__init__()

        self.num_nodes = num_nodes
        self.num_timesteps_output = num_timesteps_output

        # ASTGCN blocks
        self.blocks = nn.ModuleList()

        for i in range(nb_block):
            self.blocks.append(
                ASTGCNBlock(
                    num_nodes,
                    in_channels if i == 0 else nb_time_filter,
                    nb_chev_filter,
                    num_timesteps_input,
                    K
                )
            )

        # Output layer
        self.final_conv = nn.Conv2d(
            nb_block * nb_chev_filter,
            num_timesteps_output,
            kernel_size=(1, 1)
        )

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input (batch_size, num_timesteps_input, num_nodes, in_channels)
            laplacian: Scaled graph Laplacian (num_nodes, num_nodes)

        Returns:
            Predictions (batch_size, num_timesteps_output, num_nodes, 1)
        """
        # Transpose to (B, N, F, T)
        x = x.permute(0, 2, 3, 1)

        # Pass through blocks
        block_outputs = []
        for block in self.blocks:
            x_block = block(x, laplacian)  # (B, N, out_channels)
            block_outputs.append(x_block)

        # Concatenate block outputs
        x_concat = torch.cat(block_outputs, dim=-1)  # (B, N, nb_block * nb_chev_filter)

        # Add dimensions for conv2d: (B, C, N, 1)
        x_concat = x_concat.unsqueeze(-1).permute(0, 2, 1, 3)

        # Final convolution: (B, C, N, 1) -> (B, num_timesteps_output, N, 1)
        output = self.final_conv(x_concat)

        # Transpose to (B, num_timesteps_output, N, 1)
        output = output.permute(0, 1, 2, 3)

        return output


def create_astgcn_model(config: dict) -> ASTGCN:
    """
    Factory function to create ASTGCN model from config

    Args:
        config: Configuration dictionary

    Returns:
        ASTGCN model
    """
    return ASTGCN(
        num_nodes=config.get('num_nodes'),
        in_channels=config.get('in_channels'),
        nb_block=config.get('nb_block', 2),
        nb_chev_filter=config.get('nb_chev_filter', 64),
        nb_time_filter=config.get('nb_time_filter', 64),
        time_strides=config.get('time_strides', 1),
        num_timesteps_input=config.get('num_timesteps_input', 12),
        num_timesteps_output=config.get('num_timesteps_output', 12),
        K=config.get('K', 3)
    )
