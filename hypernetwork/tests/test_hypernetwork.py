"""Tests for HyperNetwork."""

import torch
import pytest
from src.hypernetwork.model import HyperNetwork
from src.target_network.model import TargetNetwork


@pytest.fixture
def setup():
    target = TargetNetwork([1, 16, 16, 1])
    hyper  = HyperNetwork(z_dim=4, hidden_dim=64, target_shapes=target.weight_shapes)
    return hyper, target


def test_output_count(setup):
    hyper, target = setup
    assert hyper.n_params == target.n_params


def test_forward_shapes(setup):
    hyper, target = setup
    z = torch.randn(8, 4)
    params = hyper(z)
    assert len(params) == len(target.weight_shapes)
    for (W, b), (out, inp) in zip(params, target.weight_shapes):
        assert W.shape == (8, out, inp)
        assert b.shape == (8, out)


def test_single_z(setup):
    hyper, target = setup
    z = torch.randn(4)       # unbatched
    params = hyper(z)
    assert params[0][0].shape == (target.weight_shapes[0][0], target.weight_shapes[0][1])


def test_gradients_flow(setup):
    hyper, target = setup
    z = torch.randn(4, 4)
    x = torch.randn(4, 10, 1)

    preds = []
    params = hyper(z)
    for b in range(4):
        tp = [(W[b], bv[b]) for W, bv in params]
        preds.append(target(x[b], tp))

    loss = torch.stack(preds).mean()
    loss.backward()

    for p in hyper.parameters():
        assert p.grad is not None, "Gradient did not reach hypernetwork parameter"
