from typing import Any, Callable, Tuple

from torch.utils.data import Dataset


class TransformingDataset(Dataset[Tuple[Any, Any]]):
    dataset: Dataset[Tuple[Any, Any]]
    transform: Callable[[Any], Any]
    
    def __init__(self, dataset: Dataset[Tuple[Any, Any]], transform:Callable[[Any], Any]) -> None:
        self.dataset = dataset
        self.transform = transform
        
    def append_transform(self, transform: Callable[[Any], Any]) -> None:
        self.transform = lambda x: transform(self.transform(x))
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        return self.transform(self.dataset[idx][0]), self.dataset[idx][0]
    
    # forward all the methods of the TransformingDataset to the dataset
    def __getattribute__(self, name: str) -> Any:
        if name in ["__len__", "__getitem__", "append_transform"]:
            return super().__getattribute__(name)
        else:
            return getattr(self.dataset, name)
    

