from typing import Union, Tuple, Dict, Any
from sympy import false
import shutil

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchtext import datasets as textds
from torchvision.transforms import ToTensor, Resize
import warnings
import pandas as pd
import os
from os.path import join
import json
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import numpy as np

from abc import ABC, abstractmethod

from . import CreateToriDataset
from .dataset_cloud import DatasetCloud

class AbstractDataLoader(ABC):
    """The abstractr class to interface the
    Giotto dataloaders"""
    @abstractmethod
    def build_dataloaders(self):
        pass


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


class TorchDataLoader(AbstractDataLoader):
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
        

class DataLoaderFromImages(AbstractDataLoader):
    """This class is useful to build dataloaders
    from different images you have in a folder
    
    Args:
        training_folder (string):
            The path to the folder where the training
            images are located
        test_folder (string):
            The path to the folder where the
            test images are located
        labels_file (string):
            The path and file name of the labels.
            It shall be a csv file with two columns.
            The first columns contains the name of the
            image and the second one contains the
            label value
    """
    def __init__(self, training_folder=".",
                 test_folder=".",
                 labels_file="labels.csv"):
        self.training_folder = training_folder
        self.test_folder = test_folder
        self.labels_file = labels_file
        
    def build_dataloaders(self, size: tuple=(128, 128), **kwargs) -> tuple:
        """This method builds the dataloader.

        Args:
            size (tuple):
                a tuple (h,w) to convert all images
                to the same size
            kwargs (dict):
                additional arguments to pass to the
                DataLoaders

        Returns:
            tuple of torch.DataLoader:
                the tuple with thhe training and
                test dataloader directly usable in
                the pipeline class
        """
        CWD = os.getcwd()
        df = pd.read_csv(self.labels_file)
        os.chdir(self.training_folder)  # fail if not-found
        tr_data = []
        list_of_file_names = os.listdir()
        for name in tqdm(list_of_file_names):
            try:
                image = Image.open(name)
                imageT = ToTensor()(Resize(size)(image))
                label = df[df[df.columns[0]]==name][df.columns[-1]].to_numpy()[0]
                tr_data.append((imageT, label))
                image.close()
            except:
                warnings.warn("This file could not be " +
                              "loaded due to incompatible format: " +
                              name)
            
        os.chdir(CWD)
        os.chdir(self.test_folder)  # fail if not-found
        ts_data = []
        list_of_file_names = os.listdir()
        for name in list_of_file_names:
            try:
                image = Image.open(name)
            except UnidentifiedImageError:
                warnings.warn(f"The image {name} canot be loaded. Skipping it.")
            imageT = ToTensor()(Resize(size)(image))
            ts_data.append(imageT)

        train_dataloader = DataLoader(tr_data,
                                      **kwargs)
        test_dataloader = DataLoader(ts_data,
                                     **kwargs)
        os.chdir(CWD)
        return train_dataloader, test_dataloader


class DataLoaderFromArray(AbstractDataLoader):
    """This class is useful to build dataloaders
    from a array of X and y. Tensors are also supported.

    Args:
        X_train (np.array):
            The training data
        y_train (np.array):
            The training labels
        X_val (np.array):
            The validation data
        y_val (np.array):
            The validation labels
        X_test (np.array):
            The test data
        labels_file (string):
            The path and file name of the labels.
            It shall be a csv file with two columns.
            The first columns contains the name of the
            image and the second one contains the
            label value
    """
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, 
                 X_val: np.ndarray=None, y_val: np.ndarray=None, 
                 X_test: np.ndarray=None) -> None:
        self.X_train = X_train
        if len(y_train.shape) == 1:
            self.y_train = y_train.reshape(-1, 1)
        else:
            self.y_train = y_train
        self.y_train = y_train
        if X_val is not None:
            self.X_val = X_val
            if len(y_val.shape) == 1:
                self.y_val = y_val.reshape(-1, 1)
            else:
                self.y_val = y_val
        if X_test is not None:
            self.X_test = X_test

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


    def build_dataloaders(self, **kwargs) -> tuple:
        """This method builds the dataloader.

        Args:
            kwargs (dict):
                additional arguments to pass to the
                DataLoaders

        Returns:
            tuple of torch.DataLoader:
                the tuple with the training, validation and
                test dataloader directly usable in
                the pipeline class
        """
        tr_data = [(DataLoaderFromArray._from_numpy(x).float(),
                    DataLoaderFromArray._long_or_float(y)) for x, y in zip(self.X_train, self.y_train)]
        try:
            val_data = [(DataLoaderFromArray._from_numpy(x).float(),
                        DataLoaderFromArray._long_or_float(y)) for x, y in zip(self.X_val, self.y_val)]
            #val_data = [(DataLoaderFromArray._from_numpy(x).float(),
            #             torch.tensor(y).long() if isinstance(y, np.int64) or
            #                                       ('__getitem__' in dir(y)
            #            and (isinstance(y[0], np.int32) or isinstance(y[0], np.int64))) else
            #torch.tensor(y).float()) for x, y in zip(self.X_val, self.y_val)]
        except (TypeError, AttributeError):
            val_data = None
        try:
            ts_data = [DataLoaderFromArray._from_numpy(x).float() for x in self.X_test]
        except (TypeError, AttributeError):
            ts_data = None

        train_dataloader = DataLoader(tr_data,
                                      **kwargs)
        val_dataloader = DataLoader(val_data,
                                     **kwargs)
        test_dataloader = DataLoader(ts_data,
                                     **kwargs)
        return train_dataloader, val_dataloader, test_dataloader
    
    
    
class DlBuilderFromDataCloud(AbstractDataLoader):
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

                self.dl_builder = DataLoaderFromArray(data, labels)
            elif self.dataset_metadata['data_format'] == 'numpy_array':
                data = np.load(join(self.download_directory,
                                    self.dataset_name, "data.npy"))
                labels = np.load(join(self.download_directory,
                                      self.dataset_name, "labels.npy"))
                self.dl_builder = DataLoaderFromArray(data, labels)
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