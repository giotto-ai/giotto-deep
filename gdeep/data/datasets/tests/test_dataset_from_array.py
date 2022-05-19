from gdeep.data.datasets import ImageClassificationFromFiles, \
    DataLoaderBuilder, FromArray, DlBuilderFromDataCloud

from gdeep.data.preprocessors import ToTensorImage
import os
from os.path import join
import numpy as np
import torch
import logging
from google.auth.exceptions import DefaultCredentialsError  # type: ignore

LOGGER = logging.getLogger(__name__)



def test_array():
    """test class DatasetFromArray"""
    X = np.random.rand(10,4)
    y = np.random.randint(3, size=10)
    ds = FromArray(X, y)
    dl, *_ = DataLoaderBuilder((ds,)).build(({"batch_size" : 1},))
    item = next(iter(dl))
    assert torch.norm(item[0] - torch.tensor(X[0])) < 1e-6
    assert item[1] == torch.tensor(y[0])
    assert len(dl.dataset) == 10

def test_array_tensor():
    """test class DatasetFromArray"""
    X = torch.rand(10,4)
    y = torch.randint(3, size=(10,))
    ds = FromArray(X, y)
    dl, *_ = DataLoaderBuilder((ds,)).build()
    item = next(iter(dl))
    assert torch.norm(item[0] - X[0]) < 1e-6
    assert item[1] == y[0]

