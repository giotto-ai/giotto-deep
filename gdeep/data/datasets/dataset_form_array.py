
import os
import shutil
from typing import Any, Callable, Dict, Optional, \
    Optional, Tuple, TypeVar, Union

import numpy as np
import torch

from sympy import false
from torch.utils.data import DataLoader, Dataset


Tensor = torch.Tensor


class FromArray(Dataset[Tuple[Tensor, Tensor]]):
    """This class is useful to build dataloaders
    from a array of X and y. Tensors are also supported.

    Args:
        X :
            The data. The first dimension is the datum index
        y :
            The labels, need to match the first dimension
            with the data

    """
    def __init__(self, X: Union[Tensor, np.ndarray],
                 y: Union[Tensor, np.ndarray]
                 ) -> None:
        self.X = self._from_numpy(X)
        y = self._from_numpy(y)
        self.y = self._long_or_float(y)

    @staticmethod
    def _from_numpy(X: Union[Tensor, np.ndarray]) -> Tensor:
        """this is an upgrade of ``torch.from_numpy()``
        that also allows tensor input"""
        if isinstance(X, torch.Tensor):
            return X
        return torch.from_numpy(X)

    @staticmethod
    def _long_or_float(y: Union[Tensor, np.ndarray]) -> Tensor:
        """This private method converts the labels to either
        a long tensor of a float tensor"""
        if isinstance(y, torch.Tensor):
            return y
        if isinstance(y, np.float16) or isinstance(y, np.float32) or isinstance(y, np.float64):
            return torch.tensor(y).float()
        return torch.tensor(y).long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor]:
        X, y = (self.X[idx], self.y[idx])
        return X, y
