import json
import os
import shutil
import warnings
from abc import ABC, abstractmethod
from os.path import join
from collections.abc import Iterable
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, List

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sympy import false
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm

from .build_datasets import get_dataset
from .dataset_cloud import DatasetCloud
from ..transforming_dataset import TransformingDataset


Tensor = torch.Tensor
T = TypeVar('T')

class AbstractDataLoaderBuilder(ABC):
    """The abstractr class to interface the
    Giotto dataloaders"""
    @abstractmethod
    def build_dataloaders(self):
        pass


class BuildDataLoaders(AbstractDataLoaderBuilder):
    """This class builds, out of a tuple of datasets, the
    corresponding dataloaders.

    Args:
        tuple_of_datasets :
            the tuple eith the traing, validation and test
            datasets. Also one or two elemennts are acceptable:
            they will be considered as training first and
            validation afterwards.
    """
    def __init__(self, tuple_of_datasets: Tuple[Dataset[Any], ...]) -> None:
        self.tuple_of_datasets = tuple_of_datasets
        assert len(tuple_of_datasets) <= 3, "Too many Dataset inserted: maximum 3."

    def build_dataloaders(self, *args, **kwargs) -> list:
        """This method accepts the arguments of the torch
        Dataloader and applies them when creating the
        tuple
        """
        out = [None, None, None]
        for i, dataset in enumerate(self.tuple_of_datasets):
            out[i] = DataLoader(dataset, *args, **kwargs)
        return out

