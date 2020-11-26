"""Testing for simpleNN."""
# License: GNU AGPLv3

import numpy as np
import pytest
import torch

from gdeep.neural_nets import SimpleNN

def test_SimpleNN():
    X = torch.rand(10,2)
    sn = SimpleNN()
    out = sn.forward(X)
    print(out.shape)
    assert out.shape[0] == 10 and out.shape[1] == 1
