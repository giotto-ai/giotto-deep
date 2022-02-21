import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchtext import datasets as textds
from torchvision.transforms import ToTensor, Resize
from . import CreateToriDataset
import warnings
import pandas as pd
import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import numpy as np


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


class TorchDataLoader:
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
        self. convert_to_map_dataset =  convert_to_map_dataset

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
        

class DataLoaderFromImages:
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
        """This function builds the dataloader.

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
                test dataloader directly useble in
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


class DataLoaderFromArray:
    """This class is useful to build dataloaders
    from a array of X and y

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
        self.X_val = X_val
        if len(y_train.shape) == 1:
            self.y_val = y_val.reshape(-1, 1)
        else:
            self.y_val = y_val
        self.X_test = X_test

    def build_dataloaders(self, **kwargs) -> tuple:
        """This function builds the dataloader.

        Args:
            kwargs (dict):
                additional arguments to pass to the
                DataLoaders

        Returns:
            tuple of torch.DataLoader:
                the tuple with the training, validation and
                test dataloader directly useble in
                the pipeline class
        """
        tr_data = [(torch.from_numpy(x).float(), y if isinstance(y, int) else
                    torch.tensor(y).float()) for
                    x, y in zip(self.X_train, self.y_train)]
        try:
            val_data = [(torch.from_numpy(x).float(),
                         y if isinstance(y, int) else
                         torch.tensor(y).float()) for x, y in zip(self.X_val,
                                                                  self.y_val)]
        except TypeError:
            val_data = None
        try:
            ts_data = [torch.from_numpy(x).float() for x in self.X_test]
        except TypeError:
            ts_data = None

        train_dataloader = DataLoader(tr_data,
                                      **kwargs)
        val_dataloader = DataLoader(val_data,
                                     **kwargs)
        test_dataloader = DataLoader(ts_data,
                                     **kwargs)
        return train_dataloader, val_dataloader, test_dataloader