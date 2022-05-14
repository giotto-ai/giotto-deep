from gdeep.data import DatasetImageClassificationFromFiles, \
    BuildDataLoaders, DatasetFromArray, DlBuilderFromDataCloud

from gdeep.data import PreprocessImageClassification
import os
from os.path import join
import numpy as np
import torch
import logging
from google.auth.exceptions import DefaultCredentialsError  # type: ignore

LOGGER = logging.getLogger(__name__)



def test_array():
    """test class DatasetFromArray"""
    X = np.random.rand(10,4)
    y = np.random.randint(3, size=10)
    ds = DatasetFromArray(X, y)
    dl, *_ = BuildDataLoaders((ds,)).build_dataloaders(batch_size = 1)
    item = next(iter(dl))
    assert torch.norm(item[0] - torch.tensor(X[0])) < 1e-6
    assert item[1] == torch.tensor(y[0])
    assert len(dl.dataset) == 10

def test_array_tensor():
    """test class DatasetFromArray"""
    X = torch.rand(10,4)
    y = torch.randint(3, size=(10,))
    ds = DatasetFromArray(X, y)
    dl, *_ = BuildDataLoaders((ds,)).build_dataloaders(batch_size = 1)
    item = next(iter(dl))
    assert torch.norm(item[0] - X[0]) < 1e-6
    assert item[1] == y[0]


def test_images_from_file():
    """test DatasetImageClassificationFromFiles"""
    file_path = os.path.dirname(os.path.realpath(__file__))
    ds = DatasetImageClassificationFromFiles(
        os.path.join(file_path,"img_data"),
        os.path.join(file_path,"img_data","labels.csv"),
        PreprocessImageClassification((32,32)))
    dl, *_ = BuildDataLoaders((ds,)).build_dataloaders(batch_size = 2)
    assert len(next(iter(dl))[0].shape) == 4


def test_dlbuilderfromdatacloud():
    dataset_name = "SmallDataset"
    download_directory = join("examples", "data", "DatasetCloud")

    dl_cloud_builder = DlBuilderFromDataCloud(dataset_name,
                                    download_directory)

    train_dataloader, val_dataloader, test_dataloader = \
        dl_cloud_builder.build_dataloaders(batch_size=10)
    x, y = next(iter(train_dataloader))
    assert x.shape == torch.Size([10, 5])
    assert y.shape == torch.Size([10])