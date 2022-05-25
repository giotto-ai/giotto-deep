import warnings
from typing import Any, List, Tuple, TypeVar

import torch
from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import \
    OneHotEncodedPersistenceDiagram
from torch.utils.data import Dataset

from ..abstract_preprocessing import AbstractPreprocessing
from ..transforming_dataset import TransformingDataset

Tensor = torch.Tensor
PD = OneHotEncodedPersistenceDiagram

T = TypeVar('T')

class FilterPersistenceDiagramByHomologyDimension(AbstractPreprocessing[Tuple[PD, T], Tuple[PD, T]]):
    """This class filters the persistence diagrams of a dataset by their lifetime, i.e.
    the difference between the birth and death coordinates.
    
    Here we assume that the dataset is a tuple of (persistence diagram, label) and that
    the points in the diagram are sorted by ascending lifetime. This is an invariant of
    the OneHotEncodedPersistenceDiagram class but could go wrong if the diagrams are modified
    in a way that breaks this invariant.
    
    Args:
        min_lifetime: The minimum lifetime of the points in the diagram.
        max_lifetime: The maximum lifetime of the points in the diagram.
    """
    is_fitted: bool
    homology_dimensions_to_filter_by: List[int]
    
    def __init__(self,
                 homology_dimensions_to_filter: List[int]):
        self.homology_dimensions_to_filter_by = homology_dimensions_to_filter
        self.is_fitted = False
        
    def fit_to_dataset(self, dataset: Dataset[Tuple[PD, T]]) -> None:
        """This method does nothing."""
        self.is_fitted = True
        
    def __call__(self, item: Tuple[PD, T]) -> Tuple[PD, T]:
        """Filters the persistence diagram by homology dimension.
        
        Args:
            item: A tuple of (persistence diagram, label).
        
        Returns:
            A tuple of (persistence diagram, label).
        """
        if not self.is_fitted:
            raise RuntimeError("The filter is not fitted to any dataset. "
                               "Please call fit_to_dataset() first.")
        out: PD = item[0]
        list_expanded: Tensor = torch.tensor(
            [0, 0] + [1 if d in self.homology_dimensions_to_filter_by else 0 for d in range(out.shape[1] - 2)]
        )
        
        mask = (out * list_expanded).sum(dim=1) > 0
        
        return out[mask], item[1]  # type: ignore
        

        
        
        