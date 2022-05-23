import warnings
from typing import TypeVar, Tuple, Any

import torch
from torch.utils.data import Dataset

from ..abstract_preprocessing import AbstractPreprocessing
from ..transforming_dataset import TransformingDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Tensor = torch.Tensor

T = TypeVar('T')

class Normalization(AbstractPreprocessing[Tuple[Tensor, T], Tuple[Tensor, T]]):
    """This class runs the standard normalisation on all the dimensions of
    the tensors of a dataloader. For example, in case of images where each item is of
    shape ``(C, H, W)``, the average will and the standard deviations
    will be tensors of shape ``(C, H, W)``
    """
    is_fitted: bool
    mean: Tensor
    stddev: Tensor

    def __init__(self):
        self.is_fitted = False

    def fit_to_dataset(self, dataset: Dataset[Tuple[Tensor, T]]) -> None:
        self.mean = _compute_mean_of_dataset(dataset)
        self.stddev = _compute_stddev_of_dataset(dataset, self.mean)
        self.is_fitted = True

    def __call__(self, item: Tuple[Tensor, T]) -> Tuple[Tensor, T]:
        if not self.is_fitted:
            raise RuntimeError("The normalisation is not fitted to any dataset."
                               " Please call fit_to_dataset() first.")
        if not torch.all(self.stddev > 0):
            warnings.warn("The standard deviation contains zeros! Adding 1e-7")
            self.stddev = self.stddev + 1e-7
        out = (item[0] - self.mean)/ self.stddev
        return (out, item[1])

def _compute_mean_of_dataset(dataset: Dataset[Tuple[Tensor, Any]]) -> Tensor:
    """Compute the mean of the whole dataset"""
    mean: Tensor = torch.zeros(dataset[0][0].shape, dtype=torch.float64, device=DEVICE)
    for idx in range(len(dataset)):  # type: ignore
        if idx == 0:
            mean += dataset[idx][0]
        else:
            mean = (mean * idx + dataset[idx][0]) / (idx + 1)
    return mean

def _compute_stddev_of_dataset(dataset: Dataset[Tuple[Tensor, Any]], mean: Tensor) -> Tensor:
    """Compute the stddev of the whole dataset"""
    def square_diff_from_mean(x: Tuple[Tensor, Any]) -> Tuple[Tensor, Any]:
        return (x[0] - mean) ** 2, x[1]
    mean_normalized_dataset = TransformingDataset[Tuple[Tensor, Any], Tuple[Tensor, Any]] \
        (dataset, square_diff_from_mean)
    stddev: Tensor = _compute_mean_of_dataset(mean_normalized_dataset)
    return stddev.sqrt()
