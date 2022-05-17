import json
import os
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from typing import Callable, Generic, NewType, Tuple, Union, Any, List

import jsonpickle
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchvision.transforms import Resize, ToTensor

from ..abstract_preprocessing import AbstractPreprocessing

from gdeep.utility import DEVICE
# type definition
Tensor = torch.Tensor

class ToTensorImage(AbstractPreprocessing[Any, Tuple[Tensor, Tensor]]):
    """Class to preprocess image files for classification
      tasks

          Args:
              size :
                  Desired output size. If size is a sequence like (h, w),
                  output size will be matched to this. If size is an int,
                  smaller edge of the image will be matched to this number.
                  I.e, if height > width, then image will be rescaled to
                  ``(size * height / width, size)``.

      """
    def __init__(self, size: Union[int, List[int]]) -> None:
        self.size = size
        self.is_fitted = True

    def fit_to_dataset(self, dataset:Dataset[Any]) -> None:
        pass

    def __call__(self, datum: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return ToTensor()(Resize(self.size)(datum[0])), torch.tensor(datum[1],dtype=torch.long)
