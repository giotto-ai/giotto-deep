
import os
import shutil
from typing import Any, Callable, Dict, Optional, \
    Optional, Tuple, TypeVar, Union

import numpy as np
import torch

from sympy import false
from torch.utils.data import DataLoader, Dataset

Array = np.ndarray
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
    def __init__(self, x: Union[Tensor, Array],
                 y: Union[Tensor, Array]
                 ) -> None:
        self.x = self._from_numpy(x)
        y = self._from_numpy(y)
        self.y = self._long_or_float(y)

    @staticmethod
    def _from_numpy(x: Union[Tensor, Array]) -> Tensor:
        """this is an upgrade of ``torch.from_numpy()``
        that also allows tensor input.
        """
        if isinstance(x, torch.Tensor):
            return x
        return torch.from_numpy(x)

    @staticmethod
    def _long_or_float(y: Union[Tensor, Array]) -> Tensor:
        """This private method converts the labels to either
        a long tensor of a float tensor"""
        if isinstance(y, torch.Tensor):
            return y
        if y.dtype in [np.float16, np.float32, np.float64]:
            return torch.tensor(y).float()
        return torch.tensor(y).long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor]:
        x, y = (self.x[idx], self.y[idx])
        return x, y
