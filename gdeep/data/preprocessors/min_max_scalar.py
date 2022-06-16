import warnings
from typing import Any, Tuple, TypeVar

import torch
from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import \
    OneHotEncodedPersistenceDiagram
from gdeep.utility import DEVICE
from matplotlib.pyplot import axis
from torch.utils.data import Dataset

from ..abstract_preprocessing import AbstractPreprocessing
from ..transforming_dataset import TransformingDataset

Tensor = torch.Tensor
PD = OneHotEncodedPersistenceDiagram

T = TypeVar('T')


class MinMaxScalarP(AbstractPreprocessing[Tuple[Tensor, Any], Tuple[Tensor, Any]]):
    """This class runs the standard min-max scaling on all elements of the dataset.
    
    For example. The transformation is:
    `` X_scaled = X_std * (max - min) + min `` 
    """
    is_fitted: bool
    min_max = Tuple[Tensor, Tensor]

    def __init__(self):
        self.is_fitted = False

    def fit_to_dataset(self, dataset: Dataset[Tuple[Tensor, Any]]) -> None:
        self.min_max = _compute_min_max_of_dataset(
            dataset
        )
        self.is_fitted = True

    def __call__(self, item: Tuple[Tensor, Any]) -> Tuple[Tensor, Any]:
        if not self.is_fitted:
            raise RuntimeError("The normalisation is not fitted to any dataset."
                               " Please call fit_to_dataset() first.")
        if not torch.all(self.min_max[1] > 0):  # type: ignore
            warnings.warn("The max values are zeros! Adding 1e-7")
            self.min_max = (self.min_max[0], self.min_max[1] + 1e-7)  # type: ignore
        item[0] = (item[0] - self.min_max[0]) / self.min_max[1]  # type: ignore
        return item


def _compute_min_max_of_dataset(dataset: Dataset[Tuple[Tensor, Any]]) -> Tuple[Tensor, Tensor]:
    """Compute the min and max of the whole dataset"""
    min: Tensor = dataset[0][0].clone().to(DEVICE)
    max: Tensor = dataset[0][0].clone().to(DEVICE)
    for item in dataset:   # type: ignore
        min = torch.min(min, item[0])
        max = torch.max(max, item[0])
    return min, max
