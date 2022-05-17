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
from torchtext import datasets as textds
from torchvision import datasets
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm

from .tori import CreateToriDataset
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


class DatasetNameError(Exception):
    """Exception for the improper dataset
    name"""
    pass


class MapDataset(Dataset[Any]):
    """Class to get a MapDataset from
    an iterable one.

    Args:
        data_list (list):
            the list(IterableDataset)
    """
    def __init__(self, data_list:Iterable) -> None:
        self.data_list = data_list

    def __getitem__(self, index:int ) -> Any:
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


class BuildDataLoaders(AbstractDataLoaderBuilder):
    """This class builds, out of a tuple of datasets, the
    corresponding dataloaders.

    Args:
        tuple_of_datasets (tuple of Dataset):
            the tuple eith the traing, validation and test
            datasets. Also one or two elemennts are acceptable:
            they will be considered as training first and
            validation afterwards.
    """
    def __init__(self, tuple_of_datasets: Tuple) -> None:
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


class TorchDataLoader(AbstractDataLoaderBuilder):
    """Class to obtain DataLoaders from the classical
    datasets available on pytorch.

    Args:
        name (string):
            check the available datasets at
            https://pytorch.org/vision/stable/datasets.html
        convert_to_map_dataset (bool):
            whether to convert an IterableDataset to a
            MapDataset

    """
    def __init__(self, name: str="MNIST", convert_to_map_dataset: bool=False) -> None:
        self.name = name
        self.convert_to_map_dataset = convert_to_map_dataset

    def _build_datasets(self) -> None:
        """Private method to build the dataset from
        the name of the dataset.
        """
        if self.name in ["DoubleTori", "EntangledTori", "Blobs"]:
            ctd = CreateToriDataset(self.name)
            self.training_data = ctd.generate_dataset()
            ctd2 = CreateToriDataset(self.name)
            self.test_data = ctd2.generate_dataset(n_pts=20)
        else:
            try:
                string_train = 'datasets.' + self.name + '(root="data",' + \
                    'train=True,download=True,transform=ToTensor())'
                string_test = 'datasets.' + self.name + '(root="data",' + \
                    'train=False,download=True,transform=ToTensor())'

                self.training_data = eval(string_train)
                self.test_data = eval(string_test)
            except AttributeError:
                try:
                    string_data = 'textds.' + self.name + \
                        '(split=("train","test"))'
                    self.training_data, self.test_data = eval(string_data)
                except AttributeError:
                    raise DatasetNameError(f"The dataset {self.name} is neither in the"
                                           f" texts nor images datasets")
                except TypeError:
                    string_data = 'textds.' + self.name + \
                                  '(split=("train","dev"))'
                    self.training_data, self.test_data = eval(string_data)

    def _convert(self):
        """This private method converts and IterableDataset
        to a MapDataset"""
        if isinstance(self.training_data, torch.utils.data.IterableDataset):
            self.training_data = MapDataset(list(self.training_data))

        if isinstance(self.test_data, torch.utils.data.IterableDataset):
            self.test_data = MapDataset(list(self.test_data))

    def build_dataloaders(self, **kwargs) -> tuple:
        """This method is to be called once the class
        has been initialised with the dataset name
        
        Args:
            kwargs (dict):
                The keyword arguments for the torch.DataLoader
                
        Returns:
            tuple:
                The first element is the train DataLoader
                while the second the test DataLoader
        """
        self._build_datasets()
        if self.convert_to_map_dataset:
            self._convert()
        train_dataloader = DataLoader(self.training_data,
                                      **kwargs)
        test_dataloader = DataLoader(self.test_data,
                                     **kwargs)
        return train_dataloader, test_dataloader

