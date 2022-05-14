from abc import ABC, abstractmethod
import json
import os
from typing import Generic, TypeVar
import warnings
import jsonpickle

from torch.utils.data import Dataset

T = TypeVar('T')
S = TypeVar('S')

class AbstractPreprocessing(ABC, Generic[T, S]):
    @abstractmethod
    def fit_to_dataset(self, dataset: Dataset[T]) -> None:
        pass
    
    @abstractmethod
    def __call__(self, x: T) -> S:
        pass
    
    def save_pretrained(self, path:str) -> None:
        with open(os.path.join(path, self.__class__.__name__ + ".json"), "w") as outfile:
            whole_class = jsonpickle.encode(self)  # type: ignore
            json.dump(whole_class, outfile)

    def load_pretrained(self, path:str) -> None:
        try:
            with open(os.path.join(path,self.__class__.__name__ + ".json"), "r") as infile:
                whole_class = json.load(infile)
                self = jsonpickle.decode(whole_class)  # type: ignore
        except FileNotFoundError:
            warnings.warn("The transformation file does not exist; attempting to run"
                          " the transformation anyway...")