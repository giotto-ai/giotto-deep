from dataclasses import dataclass
import json
import os
import shutil
import warnings
from abc import ABC, abstractmethod
from os.path import join
from collections.abc import Iterable
from typing import Any, Callable, Dict, Optional, Tuple, \
    TypeVar, Union, List

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
    def build(self):
        pass

@dataclass
class DataLoaderParams:
    batch_size: int
    shuffle: bool
    num_workers: int
    collate_fn: Callable[[Any], Any]
    
    def copy(self):
        return DataLoaderParams(self.batch_size, self.shuffle, self.num_workers, self.collate_fn)
    
    def update_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        return self
    
    def update_shuffle(self, shuffle: bool):
        self.shuffle = shuffle
        return self
    
    def to_dict(self):
        {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }

@dataclass
class DataLoaderParamsTuples:
    train: DataLoaderParams
    test: DataLoaderParams
    validation: Optional[DataLoaderParams] = None
        
    @staticmethod
    def default(
        collate_fn: Callable[[Any], Any],
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        return DataLoaderParamsTuples(
            DataLoaderParams(batch_size, True, num_workers, collate_fn),
            DataLoaderParams(batch_size, False, num_workers, collate_fn),
            DataLoaderParams(batch_size, False, num_workers, collate_fn)
        )
        
    def to_tuple_of_dicts(self) -> Tuple[Dict, ...]:
        if self.validation:
            return (
                self.train.to_dict(),
                self.validation.to_dict(),
                self.test.to_dict()
            )
        else:
            return (
                self.train.to_dict(),
                self.test.to_dict()
            )
        
    @staticmethod
    def from_list_of_dicts(
        list_of_kwargs: List[Dict[str, Any]]
    ) -> DataLoaderParamsTuples:
        """This method accepts the arguments of the torch
        Dataloader and applies them when creating the
        tuple

        Args:
            list_of_kwargs:
                List of dictionaries, each one being the
                kwargs for the corresponding DataLoader
        """
        assert len(list_of_kwargs) == 3, "Too many Dataset inserted: maximum 3."
        return DataLoaderParamsTuples(
            DataLoaderParams(**list_of_kwargs[0]),
            DataLoaderParams(**list_of_kwargs[1]),
            DataLoaderParams(**list_of_kwargs[2])
        )

class DataLoaderBuilder(AbstractDataLoaderBuilder):
    """This class builds, out of a tuple of datasets, the
    corresponding dataloaders. Note that this class would
    use the same parameters for all the datasets

    Args:
        tuple_of_datasets :
            the tuple eith the traing, validation and test
            datasets. Also one or two elemennts are acceptable:
            they will be considered as training first and
            validation afterwards.
    """
    def __init__(self, tuple_of_datasets: List[Dataset[Any]]) -> None:
        self.tuple_of_datasets = tuple_of_datasets
        assert len(tuple_of_datasets) <= 3, "Too many Dataset inserted: maximum 3."

    def build(self,
              parameter_tuple: Optional[DataLoaderParamsTuples]=None
              ) -> List[DataLoader[Any]]:
        """This method accepts the arguments of the torch
        Dataloader and applies them when creating the
        tuple

        Args:
            tuple_of_kwargs:
                List of dictionaries, each one being the
                kwargs for the corresponding DataLoader
        """
        if parameter_tuple is not None:
            tuple_of_kwargs = parameter_tuple.to_tuple_of_dicts()
            assert len(tuple_of_kwargs) == len(self.tuple_of_datasets), \
                "Cannot match the dataloaders and the parameters. "
            out: List = []
            for dataset, kwargs in zip(self.tuple_of_datasets, tuple_of_kwargs):
                out.append(DataLoader(dataset, **kwargs))
            out += [None] * (3 - len(out))
            return out
        else:
            out: List = []
            for i, dataset in enumerate(self.tuple_of_datasets):
                out.append(DataLoader(dataset))
            out += [None] * (3 - len(out))
            return out

