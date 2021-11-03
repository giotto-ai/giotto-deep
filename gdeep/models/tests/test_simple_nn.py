"""Testing for simpleNN."""
# License: GNU AGPLv3
import pytest
import torch
from .. import FFNet


def test_FFNet():
    X = torch.rand(10, 2)
    sn = FFNet([2, 3, 2])
    out = sn.forward(X)
    print(out.shape)
    assert out.shape[0] == 10 and out.shape[1] == 2
