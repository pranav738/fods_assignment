"""
Graph WaveNet

Reference:
Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019).
Graph WaveNet for Deep Spatial-Temporal Graph Modeling.
IJCAI 2019.

Key Features:
1. Adaptive adjacency matrix (learns hidden spatial dependencies)
2. Dilated causal convolutions for long-range temporal dependencies
3. Graph convolution for spatial modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class AdaptiveAdjacency(nn.Module):
    """
    Learn adaptive adjacency matrix from data

    This allows the model to discover hidden spatial correlations
    that may not be present in the predefined adjacency matrix.
    """

    def __init__(self, num_nodes: int, embedding_dim: int = 10):
        """
        Initialize adaptive adjacency

        Args:
            num_nodes: Number of nodes
            embedding_dim: Dimension of node embeddings
        """
        super(AdaptiveAdjacency, self).__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        # Learnable node embeddings
        self.source_embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.target_embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.source_embeddings)
        nn.init.xavier_uniform_(self.target_embeddings)

    def forward(self) -> torch.Tensor:
        """
        Compute adaptive adjacency matrix

        Returns:
            Adjacency matrix (num_nodes, num_nodes)
        """
        # Compute similarity: softmax(E1 @ E2^T)
        adj = F.softmax(
            F.relu(torch.mm(self.source_embeddings, self.target_embeddings.transpose(0, 1))),
            dim=1
        )

        return adj


class MixPropagation(nn.Module):
    """
    Mix-hop propagation layer

    Combines information from multiple diffusion steps
    """

    def __init__(self, in_channels: int, out_channels: int, num_hops: int = 2):
        """
        Initialize mix propagation layer

        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_hops: Number of diffusion hops
        """
        super(MixPropagation, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_hops = num_hops

        # Weight for each hop
        self.weights = nn.ModuleList([
            nn.Linear(in_channels, out_channels)
            for _ in range(num_hops + 1)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features (batch_size, num_nodes, in_channels)
            adj: Adjacency matrix (num_nodes, num_nodes)

        Returns:
            Output features (batch_size, num_nodes, out_channels)
        """
        batch_size, num_nodes, _ = x.shape

        # Accumulate outputs
        out = self.weights[0](x)

        # Diffusion
        x_current = x
        for k in range(1, self.num_hops + 1):
            # Propagate: A @ X
            x_current = torch.matmul(adj, x_current)

            # Weighted sum
            out = out + self.weights[k](x_current)

        return out


class DilatedCausalConv(nn.Module):
    """
    Dilated Causal Convolution

    Uses dilation to capture long-range temporal dependencies
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 dilation: int = 1):
        """
        Initialize dilated causal convolution

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor
        """
        super(DilatedCausalConv, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation

        # Padding to ensure causality (no future information)
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            padding=(0, self.padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, in_channels, num_nodes, time_steps)

        Returns:
            Output tensor (batch_size, out_channels, num_nodes, time_steps)
        """
        # Apply convolution
        out = self.conv(x)

        # Remove future padding to maintain causality
        if self.padding > 0:
            out = out[:, :, :, :-self.padding]

        return out


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with dilated causal convolutions
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 num_layers: int = 2):
        """
        Initialize TCN

        Args:
            in_channels: Input channels
            out_channels: Output channels per layer
            kernel_size: Kernel size
            num_layers: Number of layers
        """
        super(TemporalConvNet, self).__init__()

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layer_in_channels = in_channels if i == 0 else out_channels

            layers.append(
                DilatedCausalConv(
                    layer_in_channels,
                    out_channels,
                    kernel_size,
                    dilation
                )
            )

        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, in_channels, num_nodes, time_steps)

        Returns:
            Output tensor (batch_size, out_channels, num_nodes, time_steps)
        """
        for layer in self.layers:
            x = self.activation(layer(x))

        return x


class GraphWaveNetLayer(nn.Module):
    """
    Single Graph WaveNet layer

    Combines temporal convolution (TCN) and spatial convolution (GCN)
    """

    def __init__(self,
                 in_channels: int,
                 dilation_channels: int,
                 skip_channels: int,
                 end_channels: int,
                 num_nodes: int,
                 kernel_size: int = 2,
                 dilation: int = 1,
                 num_hops: int = 2):
        """
        Initialize Graph WaveNet layer

        Args:
            in_channels: Input channels
            dilation_channels: Channels for dilated convolution
            skip_channels: Channels for skip connection
            end_channels: Output channels
            num_nodes: Number of nodes
            kernel_size: Temporal kernel size
            dilation: Dilation factor
            num_hops: Number of graph diffusion hops
        """
        super(GraphWaveNetLayer, self).__init__()

        # Temporal convolution (gated)
        self.filter_conv = DilatedCausalConv(in_channels, dilation_channels, kernel_size, dilation)
        self.gate_conv = DilatedCausalConv(in_channels, dilation_channels, kernel_size, dilation)

        # Spatial convolution
        self.graph_conv_filter = MixPropagation(dilation_channels, dilation_channels, num_hops)
        self.graph_conv_gate = MixPropagation(dilation_channels, dilation_channels, num_hops)

        # Skip connection
        self.skip_conv = nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1))

        # Residual connection
        self.residual_conv = nn.Conv2d(dilation_channels, end_channels, kernel_size=(1, 1))

        self.batch_norm = nn.BatchNorm2d(end_channels)

    def forward(self,
                x: torch.Tensor,
                adj: torch.Tensor,
                adaptive_adj: Optional[torch.Tensor] = None) -> tuple:
        """
        Forward pass

        Args:
            x: Input (batch_size, in_channels, num_nodes, time_steps)
            adj: Predefined adjacency matrix
            adaptive_adj: Adaptive adjacency matrix (optional)

        Returns:
            Tuple of (residual, skip)
        """
        # Temporal convolution with gating
        filter_out = self.filter_conv(x)
        gate_out = self.gate_conv(x)

        # Transpose for graph convolution: (B, C, N, T) -> (B, T, N, C)
        filter_out = filter_out.permute(0, 3, 2, 1)
        gate_out = gate_out.permute(0, 3, 2, 1)

        # Spatial convolution on predefined adjacency
        filter_out_graph = self.graph_conv_filter(filter_out, adj)
        gate_out_graph = self.graph_conv_gate(gate_out, adj)

        # Spatial convolution on adaptive adjacency (if provided)
        if adaptive_adj is not None:
            filter_out_graph = filter_out_graph + self.graph_conv_filter(filter_out, adaptive_adj)
            gate_out_graph = gate_out_graph + self.graph_conv_gate(gate_out, adaptive_adj)

        # Transpose back: (B, T, N, C) -> (B, C, N, T)
        filter_out_graph = filter_out_graph.permute(0, 3, 2, 1)
        gate_out_graph = gate_out_graph.permute(0, 3, 2, 1)

        # Gated activation: tanh(filter) ⊙ sigmoid(gate)
        gated = torch.tanh(filter_out_graph) * torch.sigmoid(gate_out_graph)

        # Skip connection
        skip = self.skip_conv(gated)

        # Residual connection
        residual = self.residual_conv(gated)
        residual = self.batch_norm(residual)

        # Add input for residual connection
        if x.shape[1] != residual.shape[1]:
            # Adjust channels if needed
            residual = residual + self.residual_conv(x)
        else:
            residual = residual + x

        return residual, skip


class GraphWaveNet(nn.Module):
    """
    Graph WaveNet model for traffic forecasting

    Combines adaptive graph learning with dilated causal convolutions
    """

    def __init__(self,
                 num_nodes: int,
                 in_channels: int,
                 dilation_channels: int = 32,
                 skip_channels: int = 256,
                 end_channels: int = 512,
                 num_layers: int = 8,
                 kernel_size: int = 2,
                 num_hops: int = 2,
                 adaptive_embedding_dim: int = 10,
                 horizon: int = 12,
                 dropout: float = 0.3):
        """
        Initialize Graph WaveNet

        Args:
            num_nodes: Number of nodes
            in_channels: Input feature channels
            dilation_channels: Channels for dilated convolution
            skip_channels: Channels for skip connections
            end_channels: Channels for final layers
            num_layers: Number of Graph WaveNet layers
            kernel_size: Temporal kernel size
            num_hops: Number of graph diffusion hops
            adaptive_embedding_dim: Dimension for adaptive adjacency embeddings
            horizon: Prediction horizon
            dropout: Dropout rate
        """
        super(GraphWaveNet, self).__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.horizon = horizon
        self.num_layers = num_layers

        # Adaptive adjacency
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, adaptive_embedding_dim)

        # Input projection
        self.start_conv = nn.Conv2d(in_channels, dilation_channels, kernel_size=(1, 1))

        # Graph WaveNet layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i

            self.layers.append(
                GraphWaveNetLayer(
                    dilation_channels,
                    dilation_channels,
                    skip_channels,
                    dilation_channels,
                    num_nodes,
                    kernel_size,
                    dilation,
                    num_hops
                )
            )

        # Output layers
        self.skip_conv0 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.skip_conv1 = nn.Conv2d(end_channels, end_channels, kernel_size=(1, 1))
        self.output_conv = nn.Conv2d(end_channels, horizon, kernel_size=(1, 1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input (batch_size, window_size, num_nodes, in_channels)
            adj: Predefined adjacency matrix (num_nodes, num_nodes)

        Returns:
            Predictions (batch_size, horizon, num_nodes, 1)
        """
        # Transpose to (B, C, N, T)
        x = x.permute(0, 3, 2, 1)

        # Input projection
        x = self.start_conv(x)

        # Compute adaptive adjacency
        adaptive_adj = self.adaptive_adj()

        # Accumulate skip connections
        skip_outputs = []

        # Pass through layers
        for layer in self.layers:
            x, skip = layer(x, adj, adaptive_adj)
            skip_outputs.append(skip)
            x = self.dropout(x)

        # Sum skip connections
        skip_sum = torch.stack(skip_outputs, dim=0).sum(dim=0)

        # Output projection
        out = F.relu(skip_sum)
        out = self.skip_conv0(out)
        out = F.relu(out)
        out = self.skip_conv1(out)
        out = self.output_conv(out)  # (B, horizon, N, T)

        # Take last time step
        out = out[:, :, :, -1:]  # (B, horizon, N, 1)

        # Transpose to (B, horizon, N, 1)
        out = out.permute(0, 1, 2, 3)

        return out


def create_graph_wavenet_model(config: dict) -> GraphWaveNet:
    """
    Factory function to create Graph WaveNet model from config

    Args:
        config: Configuration dictionary

    Returns:
        GraphWaveNet model
    """
    return GraphWaveNet(
        num_nodes=config.get('num_nodes'),
        in_channels=config.get('in_channels'),
        dilation_channels=config.get('dilation_channels', 32),
        skip_channels=config.get('skip_channels', 256),
        end_channels=config.get('end_channels', 512),
        num_layers=config.get('num_layers', 8),
        kernel_size=config.get('kernel_size', 2),
        num_hops=config.get('num_hops', 2),
        adaptive_embedding_dim=config.get('adaptive_embedding_dim', 10),
        horizon=config.get('horizon', 12),
        dropout=config.get('dropout', 0.3)
    )
