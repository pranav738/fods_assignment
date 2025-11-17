"""
Diffusion Convolutional Recurrent Neural Network (DCRNN)

Reference:
Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018).
Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.
ICLR 2018.

Key Components:
1. Diffusion Convolution (models traffic as diffusion process on graph)
2. GRU-based recurrent architecture for temporal modeling
3. Encoder-Decoder framework for seq2seq prediction
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class DiffusionConvolution(nn.Module):
    """
    Diffusion Convolution Layer

    Models information diffusion on graph using random walks:
    f(X) = Σ_{k=0}^K (θ_k1 @ P_f^k + θ_k2 @ P_b^k) @ X

    where P_f and P_b are forward and backward transition matrices
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_diffusion_steps: int = 2,
                 bias: bool = True):
        """
        Initialize diffusion convolution

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            num_diffusion_steps: Number of diffusion steps (K)
            bias: Whether to use bias
        """
        super(DiffusionConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_diffusion_steps = num_diffusion_steps

        # Weights for forward and backward diffusion
        # For each step k, we have 2 weight matrices (forward and backward)
        self.weight_forward = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels, out_channels))
            for _ in range(num_diffusion_steps + 1)
        ])

        self.weight_backward = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels, out_channels))
            for _ in range(num_diffusion_steps + 1)
        ])

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        for weight in self.weight_forward:
            nn.init.xavier_uniform_(weight)
        for weight in self.weight_backward:
            nn.init.xavier_uniform_(weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self,
                x: torch.Tensor,
                adj_forward: torch.Tensor,
                adj_backward: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features (batch_size, num_nodes, in_channels)
            adj_forward: Forward transition matrix (num_nodes, num_nodes)
            adj_backward: Backward transition matrix (num_nodes, num_nodes)

        Returns:
            Output features (batch_size, num_nodes, out_channels)
        """
        batch_size, num_nodes, _ = x.shape

        # Accumulate diffusion results
        output = torch.zeros(batch_size, num_nodes, self.out_channels).to(x.device)

        # Current diffusion states
        x_forward = x
        x_backward = x

        for k in range(self.num_diffusion_steps + 1):
            # Forward diffusion: θ_k1 @ P_f^k @ X
            forward_output = torch.matmul(x_forward, self.weight_forward[k])
            output = output + forward_output

            # Backward diffusion: θ_k2 @ P_b^k @ X
            backward_output = torch.matmul(x_backward, self.weight_backward[k])
            output = output + backward_output

            # Update diffusion states for next step
            if k < self.num_diffusion_steps:
                x_forward = torch.matmul(adj_forward, x_forward)
                x_backward = torch.matmul(adj_backward, x_backward)

        if self.bias is not None:
            output = output + self.bias

        return output


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional Gated Recurrent Unit Cell

    Replaces matrix multiplications in GRU with diffusion convolutions
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_nodes: int,
                 num_diffusion_steps: int = 2):
        """
        Initialize DCGRU cell

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_nodes: Number of nodes
            num_diffusion_steps: Number of diffusion steps
        """
        super(DCGRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_diffusion_steps = num_diffusion_steps

        # Gates: update gate (r) and reset gate (u)
        self.diffusion_r = DiffusionConvolution(
            input_dim + hidden_dim,
            hidden_dim,
            num_diffusion_steps
        )

        self.diffusion_u = DiffusionConvolution(
            input_dim + hidden_dim,
            hidden_dim,
            num_diffusion_steps
        )

        # Candidate activation
        self.diffusion_c = DiffusionConvolution(
            input_dim + hidden_dim,
            hidden_dim,
            num_diffusion_steps
        )

    def forward(self,
                x: torch.Tensor,
                h: torch.Tensor,
                adj_forward: torch.Tensor,
                adj_backward: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input (batch_size, num_nodes, input_dim)
            h: Hidden state (batch_size, num_nodes, hidden_dim)
            adj_forward: Forward transition matrix
            adj_backward: Backward transition matrix

        Returns:
            New hidden state (batch_size, num_nodes, hidden_dim)
        """
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=2)  # (B, N, input_dim + hidden_dim)

        # Update gate: u_t = σ(diffusion([x_t, h_{t-1}]))
        u = torch.sigmoid(self.diffusion_u(combined, adj_forward, adj_backward))

        # Reset gate: r_t = σ(diffusion([x_t, h_{t-1}]))
        r = torch.sigmoid(self.diffusion_r(combined, adj_forward, adj_backward))

        # Candidate activation: c_t = tanh(diffusion([x_t, r_t ⊙ h_{t-1}]))
        combined_reset = torch.cat([x, r * h], dim=2)
        c = torch.tanh(self.diffusion_c(combined_reset, adj_forward, adj_backward))

        # New hidden state: h_t = u_t ⊙ h_{t-1} + (1 - u_t) ⊙ c_t
        h_new = u * h + (1 - u) * c

        return h_new


class DCRNNEncoder(nn.Module):
    """
    DCRNN Encoder

    Processes input sequence using stacked DCGRU layers
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_nodes: int,
                 num_layers: int = 2,
                 num_diffusion_steps: int = 2):
        """
        Initialize encoder

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_nodes: Number of nodes
            num_layers: Number of DCGRU layers
            num_diffusion_steps: Number of diffusion steps
        """
        super(DCRNNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        # Stack of DCGRU layers
        self.dcgru_layers = nn.ModuleList()

        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            self.dcgru_layers.append(
                DCGRUCell(layer_input_dim, hidden_dim, num_nodes, num_diffusion_steps)
            )

    def forward(self,
                x: torch.Tensor,
                adj_forward: torch.Tensor,
                adj_backward: torch.Tensor,
                hidden_state: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input sequence (batch_size, seq_len, num_nodes, input_dim)
            adj_forward: Forward transition matrix
            adj_backward: Backward transition matrix
            hidden_state: Initial hidden states (optional)

        Returns:
            Tuple of (output, hidden_states)
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # Initialize hidden states
        if hidden_state is None:
            hidden_state = [
                torch.zeros(batch_size, num_nodes, self.hidden_dim).to(x.device)
                for _ in range(self.num_layers)
            ]

        # Process sequence
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (B, N, input_dim)

            # Pass through each layer
            for layer_idx, dcgru_layer in enumerate(self.dcgru_layers):
                hidden_state[layer_idx] = dcgru_layer(
                    x_t,
                    hidden_state[layer_idx],
                    adj_forward,
                    adj_backward
                )
                x_t = hidden_state[layer_idx]

            outputs.append(x_t)

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (B, seq_len, N, hidden_dim)

        return output, hidden_state


class DCRNNDecoder(nn.Module):
    """
    DCRNN Decoder

    Generates predictions autoregressively
    """

    def __init__(self,
                 output_dim: int,
                 hidden_dim: int,
                 num_nodes: int,
                 num_layers: int = 2,
                 num_diffusion_steps: int = 2):
        """
        Initialize decoder

        Args:
            output_dim: Output feature dimension
            hidden_dim: Hidden state dimension
            num_nodes: Number of nodes
            num_layers: Number of DCGRU layers
            num_diffusion_steps: Number of diffusion steps
        """
        super(DCRNNDecoder, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        # Stack of DCGRU layers
        self.dcgru_layers = nn.ModuleList()

        for layer in range(num_layers):
            # Decoder input is output_dim
            layer_input_dim = output_dim if layer == 0 else hidden_dim
            self.dcgru_layers.append(
                DCGRUCell(layer_input_dim, hidden_dim, num_nodes, num_diffusion_steps)
            )

        # Output projection
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self,
                x: torch.Tensor,
                hidden_state: List[torch.Tensor],
                adj_forward: torch.Tensor,
                adj_backward: torch.Tensor,
                horizon: int) -> torch.Tensor:
        """
        Forward pass (autoregressive decoding)

        Args:
            x: Initial input (batch_size, num_nodes, output_dim)
            hidden_state: Encoder hidden states
            adj_forward: Forward transition matrix
            adj_backward: Backward transition matrix
            horizon: Number of steps to predict

        Returns:
            Predictions (batch_size, horizon, num_nodes, output_dim)
        """
        batch_size = x.shape[0]
        predictions = []

        for t in range(horizon):
            # Pass through DCGRU layers
            x_t = x

            for layer_idx, dcgru_layer in enumerate(self.dcgru_layers):
                hidden_state[layer_idx] = dcgru_layer(
                    x_t,
                    hidden_state[layer_idx],
                    adj_forward,
                    adj_backward
                )
                x_t = hidden_state[layer_idx]

            # Project to output dimension
            output_t = self.projection(x_t)  # (B, N, output_dim)
            predictions.append(output_t)

            # Use prediction as next input (autoregressive)
            x = output_t

        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # (B, horizon, N, output_dim)

        return predictions


class DCRNN(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network

    End-to-end encoder-decoder model for traffic forecasting
    """

    def __init__(self,
                 num_nodes: int,
                 input_dim: int,
                 output_dim: int = 1,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 num_diffusion_steps: int = 2,
                 horizon: int = 12):
        """
        Initialize DCRNN model

        Args:
            num_nodes: Number of nodes
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of DCGRU layers
            num_diffusion_steps: Number of diffusion steps
            horizon: Prediction horizon
        """
        super(DCRNN, self).__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        # Encoder
        self.encoder = DCRNNEncoder(
            input_dim,
            hidden_dim,
            num_nodes,
            num_layers,
            num_diffusion_steps
        )

        # Decoder
        self.decoder = DCRNNDecoder(
            output_dim,
            hidden_dim,
            num_nodes,
            num_layers,
            num_diffusion_steps
        )

    def forward(self,
                x: torch.Tensor,
                adj_forward: torch.Tensor,
                adj_backward: torch.Tensor,
                teacher_forcing_ratio: float = 0.5,
                targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input sequence (batch_size, seq_len, num_nodes, input_dim)
            adj_forward: Forward transition matrix (num_nodes, num_nodes)
            adj_backward: Backward transition matrix (num_nodes, num_nodes)
            teacher_forcing_ratio: Probability of using ground truth during training
            targets: Ground truth targets (for teacher forcing)

        Returns:
            Predictions (batch_size, horizon, num_nodes, output_dim)
        """
        # Encode input sequence
        encoder_output, hidden_state = self.encoder(x, adj_forward, adj_backward)

        # Initialize decoder input (last observed value)
        decoder_input = x[:, -1, :, :self.output_dim]  # (B, N, output_dim)

        # Decode
        predictions = self.decoder(
            decoder_input,
            hidden_state,
            adj_forward,
            adj_backward,
            self.horizon
        )

        return predictions


def create_dcrnn_model(config: dict) -> DCRNN:
    """
    Factory function to create DCRNN model from config

    Args:
        config: Configuration dictionary

    Returns:
        DCRNN model
    """
    return DCRNN(
        num_nodes=config.get('num_nodes'),
        input_dim=config.get('input_dim'),
        output_dim=config.get('output_dim', 1),
        hidden_dim=config.get('hidden_dim', 64),
        num_layers=config.get('num_layers', 2),
        num_diffusion_steps=config.get('num_diffusion_steps', 2),
        horizon=config.get('horizon', 12)
    )
