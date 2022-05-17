import json
import os
import shutil
import warnings
from abc import ABC, abstractmethod
from os.path import join
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

import numpy as np
from copy import deepcopy
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sympy import false
from torch.utils.data import DataLoader, Dataset
from torchtext import datasets as textds
from torchvision import datasets
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm



Tensor = torch.Tensor
T = TypeVar('T')


class ImageClassificationFromFiles(Dataset[Any]):
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

    img_folder: str
    labels_file: str
    def __init__(self, img_folder: str=".",
                 labels_file:str="labels.csv",
                 ) -> None:

        self.img_folder = img_folder
        self.img_labels = pd.read_csv(labels_file)


    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx:int) -> Tuple[Any, Union[str, int]]:
        image = self._get_image(idx)
        image_out = deepcopy(image)
        label = self.img_labels.iloc[idx, 1]
        image.close()
        return image_out, label

    def _get_image(self, idx: int) -> Any:
        """this method gets the i-th image in the labels.csv
        file.
        """
        img_path = os.path.join(self.img_folder, self.img_labels.iloc[idx, 0])
        try:
            image = Image.open(img_path)
        except UnidentifiedImageError:
            warnings.warn(f"The image {img_path} canot be loaded. Skipping it.")
            return None
        return image

