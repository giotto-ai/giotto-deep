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


def get_dataset(key: str, **kwargs) -> Dataset[Any]:  # type: ignore
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
    factory.register_builder("Caltech101", TorchvisionDatasetBuilder("Caltech101"))
    factory.register_builder("Caltech256", TorchvisionDatasetBuilder("Caltech256"))
    factory.register_builder("CelebA", TorchvisionDatasetBuilder("CelebA"))
    factory.register_builder("CIFAR10", TorchvisionDatasetBuilder("CIFAR10"))
    factory.register_builder("CIFAR100", TorchvisionDatasetBuilder("CIFAR100"))
    factory.register_builder("Country211", TorchvisionDatasetBuilder("Country211"))
    factory.register_builder("DTD", TorchvisionDatasetBuilder("DTD"))
    factory.register_builder("EMNIST", TorchvisionDatasetBuilder("EMNIST"))
    factory.register_builder("EuroSAT", TorchvisionDatasetBuilder("EuroSAT"))
    factory.register_builder("FakeData", TorchvisionDatasetBuilder("FakeData"))
    factory.register_builder("FashionMNIST", TorchvisionDatasetBuilder("FashionMNIST"))
    factory.register_builder("FER2013", TorchvisionDatasetBuilder("FER2013"))
    factory.register_builder("FGVCAircraft", TorchvisionDatasetBuilder("FGVCAircraft"))
    factory.register_builder("Flickr8k", TorchvisionDatasetBuilder("Flickr8k"))
    factory.register_builder("Flickr30k", TorchvisionDatasetBuilder("Flickr30k"))
    factory.register_builder("Flowers102", TorchvisionDatasetBuilder("Flowers102"))
    factory.register_builder("Food101", TorchvisionDatasetBuilder("Food101"))
    factory.register_builder("GTSRB", TorchvisionDatasetBuilder("GTSRB"))
    factory.register_builder("INaturalist", TorchvisionDatasetBuilder("INaturalist"))
    factory.register_builder("ImageNet", TorchvisionDatasetBuilder("ImageNet"))
    factory.register_builder("KMNIST", TorchvisionDatasetBuilder("KMNIST"))
    factory.register_builder("LFWPeople", TorchvisionDatasetBuilder("LFWPeople"))
    factory.register_builder("LSUN", TorchvisionDatasetBuilder("LSUN"))
    factory.register_builder("MNIST", TorchvisionDatasetBuilder("MNIST"))
    factory.register_builder("Omniglot", TorchvisionDatasetBuilder("Omniglot"))
    factory.register_builder("OxfordIIITPet", TorchvisionDatasetBuilder("OxfordIIITPet"))
    factory.register_builder("Places365", TorchvisionDatasetBuilder("Places365"))
    factory.register_builder("PCAM", TorchvisionDatasetBuilder("PCAM"))
    factory.register_builder("QMNIST", TorchvisionDatasetBuilder("QMNIST"))
    factory.register_builder("RenderedSST2", TorchvisionDatasetBuilder("RenderedSST2"))
    factory.register_builder("SEMEION", TorchvisionDatasetBuilder("SEMEION"))
    factory.register_builder("SBU", TorchvisionDatasetBuilder("SBU"))
    factory.register_builder("StanfordCars", TorchvisionDatasetBuilder("StanfordCars"))
    factory.register_builder("STL10", TorchvisionDatasetBuilder("STL10"))
    factory.register_builder("SUN397", TorchvisionDatasetBuilder("SUN397"))
    factory.register_builder("SVHN", TorchvisionDatasetBuilder("SVHN"))
    factory.register_builder("USPS", TorchvisionDatasetBuilder("USPS"))

    factory.register_builder("DoubleTori", ToriDatasetBuilder("DoubleTori"))
    factory.register_builder("Blobs", ToriDatasetBuilder("Blobs"))
    factory.register_builder("EntangledTori", ToriDatasetBuilder("EntangledTori"))

    factory.register_builder("AG_NEWS", TorchtextDatasetBuilder("AG_NEWS"))
    factory.register_builder("AmazonReviewFull", TorchtextDatasetBuilder("AmazonReviewFull"))
    factory.register_builder("AmazonReviewPolarity", TorchtextDatasetBuilder("AmazonReviewPolarity"))
    factory.register_builder("DBpedia", TorchtextDatasetBuilder("DBpedia"))
    factory.register_builder("IMDb", TorchtextDatasetBuilder("IMDb"))
    factory.register_builder("SogouNews", TorchtextDatasetBuilder("SogouNews"))
    factory.register_builder("SST2", TorchtextDatasetBuilder("SST2"))
    factory.register_builder("YahooAnswers", TorchtextDatasetBuilder("YahooAnswers"))
    factory.register_builder("YelpReviewFull", TorchtextDatasetBuilder("YelpReviewFull"))
    factory.register_builder("YelpReviewPolarity", TorchtextDatasetBuilder("YelpReviewPolarity"))
    factory.register_builder("IWSLT2016", TorchtextDatasetBuilder("IWSLT2016"))
    factory.register_builder("IWSLT2017", TorchtextDatasetBuilder("IWSLT2017"))
    factory.register_builder("Multi30k", TorchtextDatasetBuilder("Multi30k"))
    factory.register_builder("SQuAD1", TorchtextDatasetBuilder("SQuAD1"))
    factory.register_builder("SQuAD2", TorchtextDatasetBuilder("SQuAD2"))
    factory.register_builder("WikiText2", TorchtextDatasetBuilder("WikiText2"))
    factory.register_builder("WikiText103", TorchtextDatasetBuilder("WikiText103"))
    factory.register_builder("PennTreebank", TorchtextDatasetBuilder("PennTreebank"))

    return factory.build(key, **kwargs)  # type: ignore



class MapDataset(Dataset[Any]):
    """Class to get a MapDataset from
    an iterable one.

    Args:
        data_list (list):
            the list(IterableDataset)
    """
    def __init__(self, iterable_ds: torch.utils.data.IterableDataset[Any]) -> None:
        self.iterable_ds = iterable_ds

    def __getitem__(self, idx:int ) -> Any:
        return self.iterable_ds[idx]

    def __len__(self):
        return len(self.iterable_ds)


class BuildDatasets:
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
    def __init__(self, name: str="MNIST", convert_to_map_dataset:bool=False) -> None:
        self.convert_to_map_dataset = convert_to_map_dataset
        self.name = name

    def build_datasets(self, **kwargs) -> Tuple[Dataset[Any],
                                                Optional[Dataset[Any]],
                                                Optional[Dataset[Any]]]:
        """Method that returns the dataset.

        Args:
            kwargs:
                the arguments to pass to the dataset builder.
                For example, you may want to use the options
                ``split=("train","dev")`` or ``split=("train","test")``
        """
        dataset_tuple = get_dataset(self.name, **kwargs)  # type: ignore
    
        if len(dataset_tuple) == 1:
            train_ds = dataset_tuple[0]
            valid_ds = None
            test_ds = None
        if len(dataset_tuple) == 2:
            train_ds = dataset_tuple[0]
            valid_ds = dataset_tuple[1]
            test_ds = None
        if len(dataset_tuple) == 3:
            train_ds = dataset_tuple[0]
            valid_ds = dataset_tuple[1]
            test_ds = dataset_tuple[2]
        else:
            train_ds = dataset_tuple
            valid_ds = None
            test_ds = None

        if self.convert_to_map_dataset:
            train_ds, valid_ds, test_ds = self._convert(train_ds, valid_ds, test_ds)
        return train_ds, valid_ds, test_ds

    def _convert(self, training_data: Dataset[Any],
                 validation_data: Optional[Dataset[Any]]=None,
                 test_data: Optional[Dataset[Any]]=None ) -> Tuple[Dataset[Any],
                                                                   Optional[Dataset[Any]],
                                                                   Optional[Dataset[Any]]]:
        """This private method converts and IterableDataset
        to a MapDataset"""
        if isinstance(training_data, torch.utils.data.IterableDataset):
            training_data = MapDataset(list(training_data))
        if validation_data is not None:
            if isinstance(validation_data, torch.utils.data.IterableDataset):
                validation_data = MapDataset(list(validation_data))
        if test_data is not None:
            if isinstance(test_data, torch.utils.data.IterableDataset):
                test_data = MapDataset(list(test_data))
        return training_data, validation_data, test_data
