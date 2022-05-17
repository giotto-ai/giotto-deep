from typing import Any, Callable, Generic, TypeVar

from torch.utils.data import Dataset

R = TypeVar('R')
S = TypeVar('S')
T = TypeVar('T')


class TransformingDataset(Dataset[S], Generic[R, S]):
    """This class is the base class for the all
    Datasets that need to be transformed via preprocessors.
    This base class expects to get data from Dataset.

    Args:
        dataset :
            The source dataset for this class.
        transform :
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

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> S:
        """The output of this method is one element
        in the dataset. The type of this element will
        change accordingly to the dataset itself. For
        example, in case of a text classification
        dataset, the output would probably be a tuple
        like ``(label, string)``
        """
        return self.transform(self.dataset[idx])

    
    # forward all other methods of the TransformingDataset to the Dataset
    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def append_transform(dataset: TransformingDataset[R, S], transform: Callable[[S], T]) \
    -> TransformingDataset[R, T]:
    """This function allows to concatenate different preprocessors
    that the usr may want to stack.

    Args:
        dataset :
            A dataset on which it is possible to apply transforms
        transform :
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