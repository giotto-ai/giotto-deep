from typing import Any, Callable

from torch.utils.data import Dataset



class TransformingDataset(Dataset[Any]):
    dataset: Dataset[Any]
    transform: Callable[[Any], Any]
    
    def __init__(self, dataset: Dataset[Any], transform:Callable[[Any], Any]) -> None:
        self.dataset = dataset
        self.transform = transform
        
    def append_transform(self, transform: Callable[[Any], Any]) -> None:
        self.transform = lambda x: transform(self.transform(x))
    
    def __getitem__(self, idx: int):
        return self.transform(self.dataset[idx])
    
    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore