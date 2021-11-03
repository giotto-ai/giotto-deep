import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchtext import datasets as textds
from torchvision.transforms import ToTensor, Resize
from . import CreateToriDataset
import warnings
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm


class TorchDataLoader:
    """Class to obtain DataLoaders from the classical
    datasets available on pytorch

    Args:
        name (string):
            check the available datasets at
            https://pytorch.org/vision/stable/datasets.html
        n_pts (int):
            number of points in customly generated
            point clouds

    """
    def __init__(self, name="MNIST"):
        self.name = name

    def build_dataset(self):
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
                    warnings.warn("The dataset name is neither in the" +
                                  "text nor images datasets")

    def build_dataloader(self, **kwargs):
        self.build_dataset()
        train_dataloader = DataLoader(self.training_data,
                                      pin_memory=True,
                                      **kwargs)
        test_dataloader = DataLoader(self.test_data,
                                     pin_memory=True,
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
        
    def create(self, size = (128, 128), **kwargs):
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
            image = Image.open(name)
            imageT = ToTensor()(Resize(size)(image))
            ts_data.append(imageT)

        train_dataloader = DataLoader(tr_data,
                                      pin_memory=True,
                                      **kwargs)
        test_dataloader = DataLoader(ts_data,
                                     pin_memory=True,
                                     **kwargs)
        os.chdir(CWD)
        return train_dataloader, test_dataloader
