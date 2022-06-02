"""Testing for simpleNN."""
# License: GNU AGPLv3
import pytest
import torch
from .. import FFNet, PeriodicNeuralNetwork


def test_periodic():
    X = torch.rand(10, 2)
    sn = FFNet([2, 3, 2])
    bdry_list = [(-1,1),(0.5,1)]
    period = PeriodicNeuralNetwork(sn, bdry_list)
    out = period.forward(X)
    assert out.shape[0] == 10 and out.shape[1] == 2
