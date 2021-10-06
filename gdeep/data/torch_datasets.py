import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchtext import datasets as textds
from torchvision.transforms import ToTensor
from . import CreateToriDataset
import warnings


class TorchDataLoader:
    """Class to obtain DataLoaders from the classical
    datasets available on pytorch

    Args:
        name (string): check the available datasets at
            https://pytorch.org/vision/stable/datasets.html
        n_pts (int): number of points in customly generated
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
