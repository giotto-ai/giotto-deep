import json
import os
import shutil
import warnings
from abc import ABC, abstractmethod
from os.path import join
from collections.abc import Iterable
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, List

from torchtext.data import to_map_style_dataset
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sympy import false
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets
from torchtext import datasets as textdatasets

from .tori import ToriDataset
from gdeep.utility import DEFAULT_DOWNLOAD_DIR
from .dataset_cloud import DatasetCloud
from ..dataset_factory import DatasetFactory
from ..transforming_dataset import TransformingDataset

Tensor = torch.Tensor
T = TypeVar('T')


class TorchvisionDatasetBuilder(object):
    """Builder class for the torchvision dataset
        and all its variations"""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def __call__(self, **kwargs):  # type: ignore
        return datasets.__getattribute__(self.dataset_name)(  # type: ignore
            root=DEFAULT_DOWNLOAD_DIR, **kwargs  # type: ignore
        )


class TorchtextDatasetBuilder(object):
    """Builder class for the torchtext dataset
        and all its variations"""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def __call__(self, **kwargs):  # type: ignore
        return textdatasets.__getattribute__(self.dataset_name)(  # type: ignore
            root=DEFAULT_DOWNLOAD_DIR, **kwargs  # type: ignore
        )


class ToriDatasetBuilder(object):
    """Builder class for the torus dataset
    and all its variations"""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, **kwargs):  # type: ignore
        return ToriDataset(name=self.name, **kwargs)


def get_dataset(key: str, **kwargs) -> Tuple[Dataset[Any]]:
    """ Get a dataset from the factory

    Args:
        key :
            The name of the dataset, correspondiong
            to the key in the list of builders
        **kwargs:
            The keyword arguments to pass to the dataset builder

    Returns:
        torch.utils.data.Dataset:
            The dataset
    """
    factory = DatasetFactory()

    factory.register_builder("DoubleTori", ToriDatasetBuilder("DoubleTori"))
    factory.register_builder("Blobs", ToriDatasetBuilder("Blobs"))
    factory.register_builder("EntangledTori", ToriDatasetBuilder("EntangledTori"))

    for dataset_name in datasets.__all__:
        factory.register_builder(dataset_name, TorchvisionDatasetBuilder(dataset_name))

    for dataset_name in textdatasets.__all__:
        factory.register_builder(dataset_name, TorchtextDatasetBuilder(dataset_name))

    return factory.build(key, **kwargs)  # type: ignore


class DatasetBuilder:
    """Class to obtain Datasets from the classical
    datasets available on pytorch. Also the torus dataset
    and all its variations can be found here

    Args:
        name:
            check the available datasets at
            https://pytorch.org/vision/stable/datasets.html
            and https://pytorch.org/text/stable/datasets.html
        convert_to_map_dataset:
            whether to conver to the MapDataset or to keep
            IterableDataset

    """
    train_ds: Dataset[Any]
    valid_ds: Optional[Dataset[Any]]
    test_ds: Optional[Dataset[Any]]

    def __init__(self, name: str="MNIST", convert_to_map_dataset:bool=False) -> None:
        self.convert_to_map_dataset = convert_to_map_dataset
        self.name = name

    def build(self, **kwargs) -> Tuple[Dataset[Any],
                                                Optional[Dataset[Any]],
                                                Optional[Dataset[Any]]]:
        """Method that returns the dataset.

        Args:
            kwargs:
                the arguments to pass to the dataset builder.
                For example, you may want to use the options
                ``split=("train","dev")`` or ``split=("train","test")``
        """
        dataset_tuple: Tuple[Dataset[Any]] = get_dataset(self.name, **kwargs)  # type: ignore
    
        if len(dataset_tuple) == 1:
            self.train_ds = dataset_tuple[0]
            self.valid_ds = None
            self.test_ds = None
        elif len(dataset_tuple) == 2:
            self.train_ds = dataset_tuple[0]
            self.valid_ds = dataset_tuple[1]
            self.test_ds = None
        elif len(dataset_tuple) == 3:
            self.train_ds = dataset_tuple[0]
            self.valid_ds = dataset_tuple[1]
            self.test_ds = dataset_tuple[2]
        else:
            self.train_ds = dataset_tuple
            self.valid_ds = None
            self.test_ds = None

        if self.convert_to_map_dataset:
            self.train_ds, self.valid_ds, self.test_ds = self._convert(self.train_ds,
                                                                       self.valid_ds,
                                                                       self.test_ds)
        return self.train_ds, self.valid_ds, self.test_ds

    def _convert(self, training_data: Dataset[Any],
                 validation_data: Optional[Dataset[Any]]=None,
                 test_data: Optional[Dataset[Any]]=None ) -> Tuple[Dataset[Any],
                                                                   Optional[Dataset[Any]],
                                                                   Optional[Dataset[Any]]]:
        """This private method converts and IterableDataset
        to a MapDataset"""
        if isinstance(training_data, torch.utils.data.IterableDataset):
            training_data = to_map_style_dataset(training_data)
        if validation_data is not None:
            if isinstance(validation_data, torch.utils.data.IterableDataset):
                validation_data = to_map_style_dataset(validation_data)
        if test_data is not None:
            if isinstance(test_data, torch.utils.data.IterableDataset):
                test_data = to_map_style_dataset(test_data)
        return training_data, validation_data, test_data
