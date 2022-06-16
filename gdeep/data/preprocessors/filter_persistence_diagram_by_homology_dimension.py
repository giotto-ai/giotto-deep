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
    """This class filters the persistence diagrams of a dataset by their homology dimension.
    
    Here we assume that the dataset is a tuple of (persistence diagram, label) and that
    the points in the diagram are sorted by ascending lifetime. This is an invariant of
    the OneHotEncodedPersistenceDiagram class but could go wrong if the diagrams are modified
    in a way that breaks this invariant.
     
    Args:
        homology_dimensions_to_filter:
            The homology dimensions of the points in the diagram that should be kept.
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
            item: 
                A tuple of (persistence diagram, label).
        
        Returns:
            A tuple of (persistence diagram, label).
        """
        if not self.is_fitted:
            raise RuntimeError("The filter is not fitted to any dataset. "
                               "Please call fit_to_dataset() first.")
        out: List[Tensor] = []
        for homology_dimension in self.homology_dimensions_to_filter_by:
            out.append(item[0].get_all_points_in_homology_dimension(homology_dimension).get_raw_data())
        
        return PD(torch.cat(out)), item[1]
        

        
        
        