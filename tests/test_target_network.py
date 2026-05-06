"""Tests for TargetNetwork."""

import torch
import pytest
from src.target_network.model import TargetNetwork
from src.hypernetwork.model   import HyperNetwork


def make_dummy_params(target):
    """Generate random weight pairs matching target_net's shapes."""
    return [
        (torch.randn(out, inp), torch.randn(out))
        for out, inp in target.weight_shapes
    ]


def test_forward_output_shape():
    target = TargetNetwork([1, 32, 32, 1])
    params = make_dummy_params(target)
    x      = torch.randn(50, 1)
    out    = target(x, params)
    assert out.shape == (50, 1)


def test_wrong_param_count():
    target = TargetNetwork([1, 32, 32, 1])
    with pytest.raises(ValueError):
        target(torch.randn(10, 1), [])


def test_n_params_count():
    # [1,32,32,1]: (32*1+32) + (32*32+32) + (1*32+1) = 64 + 1056 + 33 = 1153
    target = TargetNetwork([1, 32, 32, 1])
    assert target.n_params == 1153


def test_different_activations():
    x = torch.randn(10, 1)
    for act in ("relu", "tanh", "silu"):
        target = TargetNetwork([1, 16, 1], activation=act)
        params = make_dummy_params(target)
        out    = target(x, params)
        assert out.shape == (10, 1)


def test_invalid_activation():
    with pytest.raises(ValueError):
        TargetNetwork([1, 16, 1], activation="sigmoid")
