import warnings
from typing import Any, Tuple, TypeVar

import torch
from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import \
    OneHotEncodedPersistenceDiagram
from torch.utils.data import Dataset

from ..abstract_preprocessing import AbstractPreprocessing
from ..transforming_dataset import TransformingDataset

Tensor = torch.Tensor
PD = OneHotEncodedPersistenceDiagram

T = TypeVar('T')


class FilterPersistenceDiagramByLifetime(AbstractPreprocessing[Tuple[PD, T], Tuple[PD, T]]):
    """This class filters the persistence diagrams of a dataset by their lifetime, i.e.
    the difference between the birth and death coordinates.
    
    Here we assume that the dataset is a tuple of (persistence diagram, label) and that
    the points in the diagram are sorted by ascending lifetime. This is an invariant of
    the OneHotEncodedPersistenceDiagram class but could go wrong if the diagrams are modified
    in a way that breaks this invariant.
    
    Args:
        min_lifetime:
            The minimum lifetime of the points in the diagram.
        max_lifetime:
            The maximum lifetime of the points in the diagram.
    """
    is_fitted: bool
    max_lifetime: float
    min_lifetime: float
    
    def __init__(self,
                 min_lifetime: float,
                 max_lifetime: float):
        self.min_lifetime = min_lifetime
        self.max_lifetime = max_lifetime
        self.is_fitted = False
        
    def fit_to_dataset(self, dataset: Dataset[Tuple[PD, T]]) -> None:
        """This method does nothing."""
        self.is_fitted = True
        
    def __call__(self, item: Tuple[PD, T]) -> Tuple[PD, T]:
        """Filters the persistence diagram by its lifetime.
        
        Args:
            item:
                A tuple of (persistence diagram, label).
        
        Returns:
            A tuple of (persistence diagram, label).
        """
        if not self.is_fitted:
            raise RuntimeError("The filter is not fitted to any dataset. "
                               "Please call fit_to_dataset() first.")
        out: Tensor = item[0].get_raw_data()
        lifetime: Tensor = out[:, 1] - out[:, 0]
        mask: Tensor = (lifetime >= self.min_lifetime) & (lifetime <= self.max_lifetime)
        out = out[mask]  # type: ignore
        return (PD(out), item[1])
    