from dataclasses import dataclass
import json
import os
import shutil
import warnings
from abc import ABC, abstractmethod
from os.path import join
from collections.abc import Iterable
from typing import Any, Callable, Dict, Generic, Optional, Sequence, Tuple, \
    TypeVar, Union, List

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm
from torch.utils.data import Sampler

from .build_datasets import get_dataset
from .dataset_cloud import DatasetCloud
from ..transforming_dataset import TransformingDataset


Tensor = torch.Tensor
T = TypeVar('T')


class AbstractDataLoaderBuilder(ABC):
    """The abstractr class to interface the
    Giotto dataloaders"""
    @abstractmethod
    def build(self, tuple_of_kwargs: List[Dict[str, Any]]):
        pass

@dataclass
class DataLoaderParams:
    batch_size: Optional[int] = 1
    shuffle: bool = False
    num_workers: int = 0
    collate_fn: Optional[Callable[[Any], Any]] = None
    batch_sampler: Optional[Sampler[Sequence]] = None
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0.0
    persistent_workers:bool = False
    
    def copy(self):
        return DataLoaderParams(**self.to_dict())
    
    def update_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        return self
    
    def update_shuffle(self, shuffle: bool):
        self.shuffle = shuffle
        return self
    
    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }

@dataclass
class DataLoaderParamsTuples:
    train: DataLoaderParams
    test: Optional[DataLoaderParams] = None
    validation: Optional[DataLoaderParams] = None
        
    @staticmethod
    def default(
        collate_fn: Callable[[Any], Any],
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0.0,
        persistent_workers:bool = False,
        with_validation: bool = False,
    ) -> "DataLoaderParamsTuples":
        dlp = DataLoaderParamsTuples(
            train=DataLoaderParams(
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                persistent_workers=persistent_workers
            ),
            test=DataLoaderParams(
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                persistent_workers=persistent_workers
            )
        )
        if with_validation:
            dlp.validation = DataLoaderParams(
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                persistent_workers=persistent_workers
            )
        return dlp
        
    def to_tuple_of_dicts(self) -> Tuple[Dict, ...]:
        if self.validation is not None and self.test is not None:
            return (
                self.train.to_dict(),
                self.validation.to_dict(),
                self.test.to_dict()
            )
        elif self.test is not None:
            return (
                self.train.to_dict(),
                self.test.to_dict()
            )
        else:
            return (
                self.train.to_dict(),
            )
        
    @staticmethod
    def from_list_of_dicts(
        list_of_kwargs: List[Dict[str, Any]]
    ) -> 'DataLoaderParamsTuples':
        """This method accepts the arguments of the torch
        Dataloader and applies them when creating the
        tuple

        Args:
            list_of_kwargs:
                List of dictionaries, each one being the
                kwargs for the corresponding DataLoader
        """
        if len(list_of_kwargs) == 3:
            return DataLoaderParamsTuples(
                DataLoaderParams(**list_of_kwargs[0]),
                DataLoaderParams(**list_of_kwargs[1]),
                DataLoaderParams(**list_of_kwargs[2])
            )
        elif len(list_of_kwargs) == 2:
            return DataLoaderParamsTuples(
                DataLoaderParams(**list_of_kwargs[0]),
                DataLoaderParams(**list_of_kwargs[1]),
                None
            )
        elif len(list_of_kwargs) == 1:
            return DataLoaderParamsTuples(
                DataLoaderParams(**list_of_kwargs[0]),
                None,
                None
            )
        else:
            raise ValueError(
                "The list of dictionaries must have 2 or 3 elements"
            )

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
              tuple_of_kwargs:Union[List[Dict[str, Any]], DataLoaderParamsTuples, None]=None
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
        if tuple_of_kwargs is None:
            for i, dataset in enumerate(self.tuple_of_datasets):
                out.append(DataLoader(dataset))
            out += [None] * (3 - len(out))
            return out
        
        if isinstance(tuple_of_kwargs, DataLoaderParamsTuples):
            tuple_of_kwargs = tuple_of_kwargs.to_tuple_of_dicts()  # type: ignore
        
        assert isinstance(tuple_of_kwargs, (list, tuple)), ("The kwargs must be a list or a tuple at"
            "this point")
        assert len(tuple_of_kwargs) == len(self.tuple_of_datasets), \
            "Cannot match the dataloaders and the parameters. "
        for dataset, kwargs in zip(self.tuple_of_datasets, tuple_of_kwargs):
            out.append(DataLoader(dataset, **kwargs))
        out += [None] * (3 - len(out))
        return out


