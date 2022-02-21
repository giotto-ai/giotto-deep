from gdeep.data import DataLoaderFromArray, TorchDataLoader
import numpy as np
import torch

def test_array():
    """test class DataLoaderFromArray"""
    X = np.random.rand(10,4)
    y = np.random.randint(3, size=10)
    X1 = np.random.rand(8, 5)
    y1 = np.random.randint(3, size=8)
    dl = DataLoaderFromArray(X, y)
    dl2 = DataLoaderFromArray(X, y, X1, y1)
    dl.build_dataloaders()
    v1, v2, v3 = dl2.build_dataloaders()
    assert torch.norm(next(iter(v1))[0] - torch.tensor(X[0])) < 1e-6
    assert next(iter(v1))[1] == torch.tensor(y[0])

def test_torchdataloader():
    dl = TorchDataLoader(name="DoubleTori")
    # train_indices = list(range(160))
    dl_tr, dl_val = dl.build_dataloaders(batch_size=23)