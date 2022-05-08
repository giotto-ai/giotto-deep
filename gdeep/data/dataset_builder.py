import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from . import CreateToriDataset
from gdeep.utility import DEFAULT_DOWNLOAD_DIR

# Define the dataset builder class for the tori dataset and torchvision datasets
# using the factory pattern
# Tori datset can be built using the CreateToriDataset class:
# dataset = CreateToriDataset(name="DoubleTori", n_points=100)
# 
# The torchvision datasets can be built using the torchvision.datasets.MNIST class:
# dataset = datasets.MNIST(root=DEFAULT_DOWNLOAD_DIR, train=True, download=True)