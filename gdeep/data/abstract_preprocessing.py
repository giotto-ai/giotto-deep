from abc import ABC, abstractmethod
import json
import os
from typing import Generic, TypeVar
import warnings
import jsonpickle

from .transforming_dataset import TransformingDataset

from torch.utils.data import Dataset

from gdeep.data.transforming_dataset import TransformingDataset

R = TypeVar("R")
S = TypeVar("S")


class AbstractPreprocessing(ABC, Generic[R, S]):
    @abstractmethod
    def fit_to_dataset(self, dataset: Dataset[R]) -> None:
        pass

    @abstractmethod
    def __call__(self, x: R) -> S:
        pass

    def transform(self, x: R) -> S:
        return self(x)

    def attach_transform_to_dataset(self, dataset: Dataset[R]) -> Dataset[S]:
        return TransformingDataset(dataset, self.transform)

    def save_pretrained(self, path: str) -> None:
        with open(
            os.path.join(path, self.__class__.__name__ + ".json"), "w"
        ) as outfile:
            whole_class = jsonpickle.encode(self)  # type: ignore
            json.dump(whole_class, outfile)

    def load_pretrained(self, path: str) -> None:
        try:
            with open(
                os.path.join(path, self.__class__.__name__ + ".json"), "r"
            ) as infile:
                whole_class = json.load(infile)
                self = jsonpickle.decode(whole_class)  # type: ignore
        except FileNotFoundError:
            warnings.warn(
                "The transformation file does not exist; attempting to run"
                " the transformation anyway..."
            )
