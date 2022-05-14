import json
import os
import shutil
import warnings
from abc import ABC, abstractmethod
from os.path import join
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

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

from . import CreateToriDataset
from .dataset_cloud import DatasetCloud
from .preprocessing_interface import AbstractPreprocessing, IdentityTransform

Tensor = torch.Tensor

class AbstractDataLoaderBuilder(ABC):
    """The abstractr class to interface the
    Giotto dataloaders"""
    @abstractmethod
    def build_dataloaders(self):
        pass

T = TypeVar('T')
class TransformableDataset(Dataset[T]):
    """This class is the interface for all
    our dataset, as it forces the API and
    thee transformations
    """
    transform: AbstractPreprocessing
    target_transform: AbstractPreprocessing
    def __init__(self,
                 transform: Optional[AbstractPreprocessing] = None,
                 target_transform: Optional[AbstractPreprocessing] = None,
                 dataset:Optional[Dataset]=None) -> None:
        if transform is None:
            self.transform = IdentityTransform()
        else:
            self.transform = transform
        if target_transform is None:
            self.target_transform = IdentityTransform()
        else:
            self.target_transform = target_transform


class BasicDataset(TransformableDataset):
    """The most basic case of datasets, i.e. those
    that have a Dataset source in which each item
    is a tuple with ``(datum, label)``"""
    def __init__(self, dataset: Dataset, transform=None, target_transform=None) -> None:
        super().__init__(transform, target_transform)
        self.dataset = dataset
        # fit the preprocessing tools
        self.transform.fit_to_data(self.dataset)
        self.target_transform.fit_to_data(self.dataset)

    def __getitem__(self, idx: int):
        return self.transform(self.dataset[idx][0]), self.target_transform(self.dataset[idx][1])

    def __len__(self) -> int:
        return len(self.dataset)


class DatasetNameError(Exception):
    """Exception for the improper dataset
    name"""
    pass


class MapDataset(Dataset):
    """Class to get a MapDataset from
    an iterable one.

    Args:
        data_list (list):
            the list(IterableDataset)
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


class BuildDataLoaders:
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
        assert len(tuple_of_datasets) <= 3, "Too many Dataset inserted"

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


class DatasetImageClassificationFromFiles(TransformableDataset):
    """This class is useful to build a dataset
    directly from image files
    
    Args:
        img_folder (string):
            The path to the folder where the training
            images are located
        labels_file (string):
            The path and file name of the labels.
            It shall be a ``.csv`` file with two columns.
            The first columns contains the name of the
            image and the second one contains the
            label value
        transform (AbstractPreprocessing):
            the instance of the class of preprocessing.
            It inherits from ``AbstractPreprocessing``
        target_transform (AbstractPreprocessing):
            the instance of the class of preprocessing.
            It inherits from ``AbstractPreprocessing``
    """
    def __init__(self, img_folder: str=".",
                 labels_file:str="labels.csv",
                 transform=None,
                 target_transform=None,
                 dataset=None  # only for pipeline compatibility
                 ) -> None:

        super().__init__(transform=transform,
                         target_transform=target_transform)
        self.img_folder = img_folder
        self.img_labels = pd.read_csv(labels_file)
        if dataset:
            self.transform.fit_to_data(dataset)
            self.target_transform.fit_to_data(dataset)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self._get_image(idx)

        label = self.img_labels.iloc[idx, 1]
        imageT = self.transform(image)
        label = self.target_transform(label)
        image.close()
        return imageT, label

    def _get_image(self, idx:int):
        img_path = os.path.join(self.img_folder, self.img_labels.iloc[idx, 0])
        try:
            image = Image.open(img_path)
        except UnidentifiedImageError:
            warnings.warn(f"The image {img_path} canot be loaded. Skipping it.")
            return None, None
        return image


class DatasetFromArray(TransformableDataset):
    """This class is useful to build dataloaders
    from a array of X and y. Tensors are also supported.

    Args:
        X (np.array or torch.Tensor):
            The data. The first dimension is the datum index
        y (np.array or torch.Tensor):
            The labels, need to match the first dimension
            with the data

    """
    def __init__(self, X: Union[Tensor, np.ndarray],
                 y: Union[Tensor, np.ndarray],
                 transform=None, target_transform=None,
                 dataset=None # for compatiblitiy reasons
                 ) -> None:
        super().__init__(transform=transform,
                         target_transform=target_transform)
        self.X = self._from_numpy(X)
        y = self._from_numpy(y)
        self.y = self._long_or_float(y)
        self.transform.fit_to_data(X)
        self.target_transform.fit_to_data(y)

    @staticmethod
    def _from_numpy(X):
        """this is torch.from_numpy() that also allows
        for tensors"""
        if isinstance(X, torch.Tensor):
            return X
        return torch.from_numpy(X)

    @staticmethod
    def _long_or_float(y):
        if isinstance(y, torch.Tensor):
            return y
        if isinstance(y, np.float16) or isinstance(y, np.float32) or isinstance(y, np.float64):
            return torch.tensor(y).float()
        return torch.tensor(y).long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor]:
        y = self.target_transform(self.y[idx])
        X = self.transform(self.X[idx])
        return X, y


class DlBuilderFromDataCloud(AbstractDataLoaderBuilder):
    """Class that loads data from Google Cloud Storage
    
    This class is useful to build dataloaders from a dataset stored in
    the GDeep Dataset Cloud on Google Cloud Storage.

    The constructor takes the name of a dataset as a string, and a string
    for the download directory. The constructor will download the dataset
    to the download directory. The dataset is downloaded in the version
    used by Datasets Cloud, which may be different from the version
    used by the dataset's original developers.
    
    Args:
        dataset_name (str):
            The name of the dataset.
        download_dir (str):
            The directory where the dataset will be downloaded.
        use_public_access (bool):
            Whether to use public access. If you want
            to use the Google Cloud Storage API, you must set this to True.
            Please make sure you have the appropriate credentials.
        path_to_credentials (str):
            Path to the credentials file.
            Only used if public_access is False and credentials are not
            provided. Defaults to None.
        

    Returns:
        torch.utils.data.DataLoader: The dataloader for the dataset.

    Raises:
        ValueError:
            If the dataset_name is not a valid dataset that exists
            in Datasets Cloud.
        ValueError:
            If the download_directory is not a valid directory.
    """
    def __init__(self,
                 dataset_name: str,
                 download_directory: str,
                 use_public_access: bool=True,
                 path_to_credentials: Union[None, str] = None,
                 ):
        self.dataset_name = dataset_name
        self.download_directory = download_directory
        
        # Download the dataset if it does not exist
        self.download_directory
        
        self._download_dataset(use_public_access=use_public_access, 
                               path_to_credentials = path_to_credentials)

        self.dl_builder = None
        
        # Load the metadata of the dataset
        with open(join(self.download_directory, self.dataset_name,
                       "metadata.json")) as f:
            self.dataset_metadata = json.load(f)
            
        # Load the data and labels of the dataset
        if self.dataset_metadata['data_type'] == 'tabular':
            if self.dataset_metadata['data_format'] == 'pytorch_tensor':
                data = torch.load(join(self.download_directory, 
                                       self.dataset_name, "data.pt"))
                labels = torch.load(join(self.download_directory, 
                                         self.dataset_name, "labels.pt"))

                self.dl_builder = BuildDataLoaders((DatasetFromArray(data, labels),))
            elif self.dataset_metadata['data_format'] == 'numpy_array':
                data = np.load(join(self.download_directory,
                                    self.dataset_name, "data.npy"))
                labels = np.load(join(self.download_directory,
                                      self.dataset_name, "labels.npy"))
                self.dl_builder = BuildDataLoaders((DatasetFromArray(data, labels),))
            else:
                raise ValueError("Data format {}"\
                    .format(self.dataset_metadata['data_format']) +
                                 "is not yet supported.")
        else:
            raise ValueError("Dataset type {} is not yet supported."\
                .format(self.dataset_metadata['data_type']))

    def _download_dataset(self,
                          path_to_credentials: Union[None, str] =None,
                          use_public_access: bool=True,) -> None:
        """Only download if the download directory does not exist already
        and if download directory contains at least three files (metadata,
        data, labels).
        
        Args:
            path_to_credentials (str):
                Path to the credentials file.
            use_public_access (bool):
                Whether to use public access. If you want
                to use the Google Cloud Storage API, you must set this to True.
                
        Returns:
            None
        """
        if (not os.path.isdir(join(self.download_directory, self.dataset_name))
                            or len(os.listdir(join(self.download_directory,
                                            self.dataset_name))) < 3):
            # Delete the download directory if it exists but does not contain
            # the wanted number of files
            if (os.path.isdir(join(self.download_directory, self.dataset_name))
                and
                len(os.listdir(join(self.download_directory,
                                   self.dataset_name))) < 3): # type: ignore
                print("Deleting the download directory because it does "+
                      "not contain the dataset")
                shutil.rmtree(self.download_directory, ignore_errors=True)
                
            print("Downloading dataset {} to {}"\
                    .format(self.dataset_name, self.download_directory))
            dataset_cloud = DatasetCloud(self.dataset_name,
                                    download_directory=self.download_directory,
                                    path_to_credentials=path_to_credentials,
                                    use_public_access = use_public_access,
                                    )
            dataset_cloud.download()
            del dataset_cloud
        else:
            print("Dataset '%s' already downloaded" % self.dataset_name)

    def get_metadata(self) -> Dict[str, Any]:
        """ Returns the metadata of the dataset.
        
        Returns:
            Dict[str, Any]:
                The metadata of the dataset.
        """
        return self.dataset_metadata
    
    def build_dataloaders(self, **kwargs)\
        -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Builds the dataloaders for the dataset.
        
        Args:
            **kwargs: Arguments for the dataloader builder.
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]:
                The dataloaders for the dataset (train, validation, test).
        """
        return self.dl_builder.build_dataloaders(**kwargs) # type: ignore
