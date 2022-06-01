import warnings
from typing import Any, Tuple, TypeVar

import torch
from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import \
    OneHotEncodedPersistenceDiagram
from torch.utils.data import Dataset

from ..abstract_preprocessing import AbstractPreprocessing
from ..transforming_dataset import TransformingDataset

from .normalization import _compute_mean_of_dataset

Tensor = torch.Tensor
PD = OneHotEncodedPersistenceDiagram

T = TypeVar('T')


class NormalizationPersistenceDiagram(AbstractPreprocessing[Tuple[PD, T], Tuple[PD, T]]):
    """This class runs the standard normalisation on the birth and death coordinates
    of the persistence diagrams of a dataset accross all the homology dimensions.
    
    The one-hot encoded persistence diagrams are kept as is.
    """
    is_fitted: bool
    mean: Tensor
    stddev: Tensor
    num_homology_dimensions: int

    def __init__(self,
                 num_homology_dimensions: int):
        self.is_fitted = False
        self.num_homology_dimensions = num_homology_dimensions

    def fit_to_dataset(self, dataset: Dataset[Tuple[PD, T]]) -> None:
        # compute the mean and set the last entries to zero
        self.mean = _compute_mean_of_dataset(
            TransformingDataset(dataset, lambda x: (x[0].get_raw_data().mean(dim=0), x[1]))
            )
        self.mean = self.mean * torch.tensor([1.0, 1.0] + [0.0] * (self.num_homology_dimensions))
        self.stddev = _compute_mean_of_dataset(
            TransformingDataset(dataset, lambda x: (((x[0].get_raw_data() - self.mean)**2).mean(dim=0), x[1]))
            )
        self.stddev = (self.stddev 
                       * torch.tensor([1.0, 1.0] + [0.0] * (self.num_homology_dimensions))
                       + torch.tensor([0.0, 0.0] + [1.0] * (self.num_homology_dimensions))
                       )
        self.stddev = torch.sqrt(self.stddev)
        self.is_fitted = True

    def __call__(self, item: Tuple[PD, T]) -> Tuple[PD, T]:
        if not self.is_fitted:
            raise RuntimeError("The normalisation is not fitted to any dataset."
                               " Please call fit_to_dataset() first.")
        if not torch.all(self.stddev > 0):
            warnings.warn("The standard deviation contains zeros! Adding 1e-7")
            self.stddev = self.stddev + 1e-7
        out: PD = PD(((item[0]._data - self.mean) / self.stddev).float())  # type: ignore
        return (out, item[1])
