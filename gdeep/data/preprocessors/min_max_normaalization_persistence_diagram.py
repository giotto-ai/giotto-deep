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
from .min_max_scalar import _compute_min_max_of_dataset

Tensor = torch.Tensor
PD = OneHotEncodedPersistenceDiagram

T = TypeVar('T')


class MinMaxScalarPersistenceDiagram(AbstractPreprocessing[Tuple[Tensor, T], Tuple[Tensor, T]]):
    """This class runs the standard min-max normalisation on the birth and death times 
    of the persistence diagrams. For example. The transformation is:
    X_scaled = X_std * (max - min) + min
    """
    is_fitted: bool
    min_max = Tuple[Tensor, Tensor]
    
    def __init__(self):
        self.is_fitted = False
        
    def fit_to_dataset(self, dataset: Dataset[Tuple[Tensor, T]]) -> None:
        self.min_max = _compute_min_max_of_dataset(
            TransformingDataset(dataset, lambda x: (torch.cat([x[0].min(), x[0].max()]), x[1]))
        )
        self.is_fitted = True
        
    def __call__(self, item: Tuple[Tensor, T]) -> Tuple[Tensor, T]:
        if not self.is_fitted:
            raise RuntimeError("The normalisation is not fitted to any dataset."
                               " Please call fit_to_dataset() first.")
        if not torch.all(self.min_max[1] > 0):  # type: ignore
            warnings.warn("The max values are zeros! Adding 1e-7")
            self.min_max = (self.min_max[0], self.min_max[1] + 1e-7)  # type: ignore
        out = (item[0] - self.min_max[0])/ self.min_max[1]  # type: ignore
        return (out, item[1])
    
    
