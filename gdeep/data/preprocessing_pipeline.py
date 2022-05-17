from typing import Any, Callable, Iterable, TypeVar, Generic

from torch.utils.data import Dataset

from .abstract_preprocessing import AbstractPreprocessing
from .transforming_dataset import TransformingDataset, append_transform

T = TypeVar('T')
R = TypeVar('R')
S = TypeVar('S')

class PreprocessingPipeline(AbstractPreprocessing[S, Any]):
    """ Pipeline to fit non-fitted preprocessors to a dataset in a sequential manner.
    The fitted preprocessing transform can be attached to a dataset using
    the ´attach_transform_to_dataset´ method.
    The intended use case is to fit the preprocessors to the training dataset and
    then attach the fitted transform to the training, validation and test datasets.

    The transform is only applied to the data and not the labels.

    Examples::

        from gdeep.data.preeprocessors import PreprocessingPipeline, Normalization, \
            PreprocessImageClassification
        from gdeep.data.datasets import DatasetImageClassificationFromFiles

        image_dataset = DatasetImageClassificationFromFiles(
            os.path.join(file_path, "img_data"),
            os.path.join(file_path, "img_data", "labels.csv"))

        preprocessing_pipeline = PreprocessingPipeline((PreprocessImageClassification((32, 32)),
                                                        Normalization()))
        preprocessing_pipeline.fit_to_dataset(image_dataset)  # this will not change the image_dataset
        preprocessed_dataset = preprocessing_pipeline.attach_transform_to_dataset(image_dataset)

    """

    transform: Callable[[S], Any]
    preprocessors: Iterable[AbstractPreprocessing[S, Any]]

    def __init__(self, preprocessors: Iterable[AbstractPreprocessing[S, Any]]) -> None:
        self.preprocessors = preprocessors
        self.is_fitted = False

    def attach_transform_to_dataset(self, dataset: Dataset[Any]) -> TransformingDataset[Any, Any]:
        return TransformingDataset(dataset, self.transform)

    def fit_to_dataset(self, dataset: Dataset[Any]) -> None:
        self.transform: Callable[[Any], Any] = lambda x: x  # type: ignore
        transformed_dataset = TransformingDataset(dataset, self.transform)
        for preprocessor in self.preprocessors:
            preprocessor.fit_to_dataset(transformed_dataset)
            transformed_dataset = append_transform(transformed_dataset,
                                                   preprocessor)
        self.transform = transformed_dataset.transform 
        self.is_fitted = True

    def __call__(self, x:S) -> Any:
        if not self.is_fitted:
            self.load_pretrained(".")
        return self.transform(x)