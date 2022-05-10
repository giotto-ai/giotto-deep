# %%
import json
import jsonpickle
from typing import Callable, Generic, Iterator, List, TypeVar

from gdeep.data import 




T = TypeVar('T')

class Pipeline(Generic[T]):
    _transformations: List[Callable[[T], T]]
    
    def __init__(self, transformations: List[Callable[[T], T]] = None) -> None:
        if transformations is None:
            self._transformations = []
        else:
            self._transformations = transformations
    
    def register(self, transformation: Callable[[T], T]) -> None:
        self._transformations.append(transformation)
        
    def __call__(self, data: T) -> T:
        for transformation in self._transformations:
            data = transformation(data)
        return data
    
    def __len__(self) -> int:
        return len(self._transformations)
    
    def __getitem__(self, index: int) -> Callable[[T], T]:
        return self._transformations[index]
    
    def __iter__(self) -> Iterator[Callable[[T], T]]:
        return iter(self._transformations)
    
    def __repr__(self) -> str:
        return f'Pipeline({self._transformations})'
    
    def __add__(self, other: 'Pipeline[T]') -> 'Pipeline[T]':
        return Pipeline(self._transformations + other._transformations)
    
    def save_to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json_transformation = jsonpickle.encode(self)
            json.dump(json_transformation, f)
            
    @classmethod
    def load_from_json(cls, path: str) -> 'Pipeline[T]':
        with open(path, 'r') as f:
            transformations = json.load(f)
        transform: Pipeline[T] = jsonpickle.decode(transformations)
        return transform



def pipeline(*transformations: Callable[[T], T]) -> Pipeline[T]:
    """
    Creates a pipeline that applies the given transformations in order.
    """
    pipeline: Pipeline[T] = Pipeline()
    for transformation in transformations:
        pipeline.register(transformation)
    return pipeline

# Sample usage:
import torch
Tensor = torch.Tensor

def add_one(x: Tensor) -> Tensor:
    return x + 1
def multiply_by_two(x: Tensor) -> Tensor:
    return x * 2

pipe: Pipeline[Tensor] = pipeline(add_one, multiply_by_two)

pipe.save_to_json('pipeline.json')
del pipe
pipe = Pipeline.load_from_json('pipeline.json')

x = torch.tensor([[1, 2], [3, 4]])
y = pipe(x)
print(y)
# %%
