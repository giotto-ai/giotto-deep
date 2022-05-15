from typing import Any, Callable, Generic, TypeVar

from torch.utils.data import Dataset

R = TypeVar('R')
S = TypeVar('S')
T = TypeVar('T')

class TransformingDataset(Dataset[S], Generic[R, S]):
    dataset: Dataset[R]
    transform: Callable[[R], S]
    
    def __init__(self,
                 dataset: Dataset[R],
                 transform: Callable[[R], S]) -> None:
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx: int) -> S:
        return self.transform(self.dataset[idx])
    
    # forward all other methods of the TransformingDataset to the dataset
    def __getattribute__(self, name: str) -> Any:
        if name in ["__init__", "__getitem__"]:
            return self.name
        return getattr(self.dataset, name)

def append_transform(dataset: TransformingDataset[R, S], transform: Callable[[S], T]) \
    -> TransformingDataset[R, T]:
    def new_transform(x: R) -> T:
        return transform(dataset.transform(x))
    return TransformingDataset(dataset.dataset, new_transform)