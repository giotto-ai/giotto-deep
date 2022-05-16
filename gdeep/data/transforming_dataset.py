from typing import Any, Callable, Generic, TypeVar, Optional

from torch.utils.data import Dataset

R = TypeVar('R')
S = TypeVar('S')
T = TypeVar('T')


class TransformingDataset(Dataset[S], Generic[R, S]):
    """This class is the base class for the all
    Datasets that need to be transformed via preprocessors.
    This base class expects to get data from Dataset.

    Args:
        dataset (torch.utils.data.Dataset):
            The source dataset for this class.
        transform (Callable):
            This is either a function defined by the
            users or a fitted preprocessor. The
            preprocessors inherits from ``AbstractPreprocessing``
    """

    dataset: Dataset[R]
    transform: Callable[[R], S]
    
    def __init__(self,
                 dataset: Dataset[R],
                 transform: Callable[[R], S]) -> None:
        self.dataset = dataset
        self.transform = transform

    
    def __getitem__(self, idx: int) -> S:
        """The output of this method is one element
        in the dataset. The type of this element will
        change accordingly to the dataset itself. For
        example, in case of a text classification
        dataset, the output would probably be a tuple
        like ``(label, string)``
        """
        return self.transform(self.dataset[idx])

    def __len__(self) -> int:
        """This method returs the length
        of the dataset"""
        if self.dataset:
            return len(self.dataset)
    
    # forward all other methods of the TransformingDataset to the Dataset
    def __getattr__(self, name: str) -> Any:
        if self.dataset:
            return getattr(self.dataset, name)


class IdentityTransformingDataset(TransformingDataset[R, R], Generic[R]):
    """This calss is the same as TransformingDataset except
    that it does not require to specify
    any transformation
    """
    def __init__(self, dataset: Dataset[R]) -> None:
        super().__init__(dataset, lambda x: x)


def append_transform(dataset: TransformingDataset[R, S], transform: Callable[[S], T]) \
    -> TransformingDataset[R, T]:
    """This function allows to concatenate different preprocessors
    that the usr may want to stack.

    Args:
        dataset (TransformingDataset):
            A dataset on which it is possible to apply transforms
        transform (Callable):
            A callable that can be applied to each item in the
            ``dataset``

    Returns:
        TransformingDataset:
            A new TransformingDataset that has been transformed
            by the transformation
    """
    def new_transform(x: R) -> T:
        return transform(dataset.transform(x))
    return TransformingDataset(dataset.dataset, new_transform)