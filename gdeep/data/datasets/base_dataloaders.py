
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, List

import torch
from torch.utils.data import DataLoader, Dataset


Tensor = torch.Tensor
T = TypeVar('T')


class AbstractDataLoaderBuilder(ABC):
    """The abstractr class to interface the
    Giotto dataloaders"""
    @abstractmethod
    def build(self, tuple_of_kwargs: List[Dict[str, Any]]):
        pass


class DataLoaderBuilder(AbstractDataLoaderBuilder):
    """This class builds, out of a tuple of datasets, the
    corresponding dataloaders. Note that this class would
    use the same parameters for all the datasets

    Args:
        tuple_of_datasets :
            the tuple eith the traing, validation and test
            datasets. Also one or two elemennts are acceptable:
            they will be considered as training first and
            validation afterwards.
    """
    def __init__(self, tuple_of_datasets: List[Dataset[Any]]) -> None:
        self.tuple_of_datasets = tuple_of_datasets
        assert len(tuple_of_datasets) <= 3, "Too many Dataset inserted: maximum 3."

    def build(self,
              tuple_of_kwargs: Optional[List[Dict[str, Any]]] = None
              ) -> List[DataLoader[Any]]:
        """This method accepts the arguments of the torch
        Dataloader and applies them when creating the
        tuple

        Args:
            tuple_of_kwargs:
                List of dictionaries, each one being the
                kwargs for the corresponding DataLoader
        """
        out: List = []
        if tuple_of_kwargs:
            assert len(tuple_of_kwargs) == len(self.tuple_of_datasets), \
                "Cannot match the dataloaders and the parameters. "

            for dataset, kwargs in zip(self.tuple_of_datasets, tuple_of_kwargs):
                out.append(DataLoader(dataset, **kwargs))
            out += [None] * (3 - len(out))
            return out
        else:
            for i, dataset in enumerate(self.tuple_of_datasets):
                out.append(DataLoader(dataset))
            out += [None] * (3 - len(out))
            return out

