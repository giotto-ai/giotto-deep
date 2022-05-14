from typing import Any, Callable, Iterable

from torch.utils.data import Dataset

from .abstract_preprocessing import AbstractPreprocessing
from .transforming_dataset import TransformingDataset


class PreprocessingPipeline:
    transform = Callable[[Any], Any]
    def __init__(self, preprocessors: Iterable[AbstractPreprocessing[Any, Any]]) -> None:
        self.preprocessors = preprocessors
    
    def attach_transform_to_dataset(self, dataset: Dataset[Any]) -> TransformingDataset:
        return TransformingDataset(dataset, self.transform)

    def fit_to_dataset(self, dataset: Dataset[Any]) -> None:
        def id_transform(x: Any) -> Any:
            return x
        self.transform = id_transform
        transformed_dataset = TransformingDataset(dataset, self.transform)
        for preprocessor in self.preprocessors:
            preprocessor.fit_to_dataset(transformed_dataset)
            transformed_dataset.append_transform(preprocessor)
        self.transform = transformed_dataset.transform