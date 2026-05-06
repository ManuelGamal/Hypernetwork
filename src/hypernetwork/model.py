"""
HyperNetwork
============
A network that generates the weights of another (target) network.

Given a task embedding z ∈ R^z_dim, the hypernetwork outputs a flat
parameter vector θ that is subsequently injected into a TargetNetwork.

Architecture
------------
z → Linear → ReLU → Linear → ReLU → Linear → θ
                                          ↓
                               (reshape into target weight matrices)
"""

import torch
import torch.nn as nn
from typing import List


class HyperNetwork(nn.Module):
    """
    Generates all parameters of a target network from a task embedding.

    Args:
        z_dim:          Dimensionality of the task/context embedding.
        hidden_dim:     Width of the hypernetwork's hidden layers.
        target_shapes:  List of (out, in) tuples describing each weight
                        matrix in the target network.  Bias vectors are
                        generated automatically (shape = (out,)).
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        target_shapes: List[tuple],
    ):
        super().__init__()

        self.target_shapes = target_shapes

        # Total number of scalars the hypernetwork must output
        # = sum of all weight elements + sum of all bias elements
        self.n_weights = sum(o * i for o, i in target_shapes)
        self.n_biases  = sum(o     for o, _ in target_shapes)
        self.n_params  = self.n_weights + self.n_biases

        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_params),
        )

        self._init_weights()

    def _init_weights(self):
        """Small-scale init keeps generated weights near zero at the start."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> List[tuple]:
        """
        Args:
            z: Task embedding of shape (batch, z_dim) or (z_dim,).

        Returns:
            List of (weight, bias) tensor pairs, one per layer in the
            target network.  Each weight has shape (out, in) and each
            bias has shape (out,).
        """
        flat = self.net(z)          # (batch, n_params) or (n_params,)
        return self._unpack(flat)

    def _unpack(self, flat: torch.Tensor) -> List[tuple]:
        """Split the flat parameter vector into per-layer (W, b) pairs."""
        params = []
        cursor = 0

        for out_dim, in_dim in self.target_shapes:
            w_size = out_dim * in_dim
            W = flat[..., cursor : cursor + w_size]
            W = W.reshape(*flat.shape[:-1], out_dim, in_dim)
            cursor += w_size

            b = flat[..., cursor : cursor + out_dim]
            cursor += out_dim

            params.append((W, b))

        return params
