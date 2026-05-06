"""
TargetNetwork
=============
A small MLP whose weights are *injected* from outside (by the HyperNetwork)
rather than learned via backprop directly.

This means the TargetNetwork itself has NO nn.Parameter objects — it is a
pure computation graph that accepts external weight tensors.
"""

import torch
import torch.nn.functional as F
from typing import List


class TargetNetwork:
    """
    Stateless MLP evaluated with externally provided weights.

    Args:
        layer_sizes: List of integers [in, h1, h2, ..., out].
                     E.g. [1, 32, 32, 1] gives a 3-layer MLP.
        activation:  'relu' | 'tanh' | 'silu'  (applied between hidden layers)
    """

    ACTIVATIONS = {
        "relu": F.relu,
        "tanh": torch.tanh,
        "silu": F.silu,
    }

    def __init__(self, layer_sizes: List[int], activation: str = "tanh"):
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(self.ACTIVATIONS)}")

        self.layer_sizes = layer_sizes
        self.activation  = self.ACTIVATIONS[activation]

        # Shapes that the HyperNetwork must produce: (out, in) per layer
        self.weight_shapes = [
            (layer_sizes[i + 1], layer_sizes[i])
            for i in range(len(layer_sizes) - 1)
        ]

    def forward(self, x: torch.Tensor, params: List[tuple]) -> torch.Tensor:
        """
        Run a forward pass using externally provided weights.

        Args:
            x:      Input tensor of shape (batch, in_dim).
            params: List of (W, b) pairs from HyperNetwork.forward().
                    Length must equal len(self.weight_shapes).

        Returns:
            Output tensor of shape (batch, out_dim).
        """
        if len(params) != len(self.weight_shapes):
            raise ValueError(
                f"Expected {len(self.weight_shapes)} weight pairs, got {len(params)}"
            )

        h = x
        for i, (W, b) in enumerate(params):
            # W: (out, in)  b: (out,)
            h = h @ W.T + b
            # Apply activation after every layer except the last
            if i < len(params) - 1:
                h = self.activation(h)

        return h

    # Convenience alias so TargetNetwork instances are callable
    def __call__(self, x: torch.Tensor, params: List[tuple]) -> torch.Tensor:
        return self.forward(x, params)

    @property
    def n_params(self) -> int:
        """Total number of scalar parameters in the target network."""
        return sum(o * i + o for o, i in self.weight_shapes)
