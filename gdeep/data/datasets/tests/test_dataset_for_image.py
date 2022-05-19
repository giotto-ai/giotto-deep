from gdeep.data.datasets import ImageClassificationFromFiles, \
    DataLoaderBuilder, DlBuilderFromDataCloud

from gdeep.data import TransformingDataset
from gdeep.data.preprocessors import ToTensorImage
import os
from os.path import join
import numpy as np
import torch
import logging
from google.auth.exceptions import DefaultCredentialsError  # type: ignore

LOGGER = logging.getLogger(__name__)



def test_images_from_file():
    """test DatasetImageClassificationFromFiles"""
    file_path = os.path.dirname(os.path.realpath(__file__))
    transform = ToTensorImage((32,32))  # this is already fitted
    ds = ImageClassificationFromFiles(
        os.path.join(file_path,"img_data"),
        os.path.join(file_path,"img_data","labels.csv"))

    tds = TransformingDataset(ds, transform)
    dl, *_ = DataLoaderBuilder((tds,)).build(({"batch_size" : 2},))
    assert len(next(iter(dl))[0].shape) == 4


def test_dlbuilderfromdatacloud():
    dataset_name = "SmallDataset"
    download_directory = join("examples", "data", "DatasetCloud")

    dl_cloud_builder = DlBuilderFromDataCloud(dataset_name,
                                    download_directory)

    train_dataloader, val_dataloader, test_dataloader = \
        dl_cloud_builder.build(({"batch_size" : 10},))
    x, y = next(iter(train_dataloader))
    assert x.shape == torch.Size([10, 5])
    assert y.shape == torch.Size([10])