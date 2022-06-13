from typing import Any, Callable, Generic, Iterable, TypeVar

from torch.utils.data import Dataset

from .abstract_preprocessing import AbstractPreprocessing
from .transforming_dataset import TransformingDataset, append_transform

T = TypeVar("T")


class PreprocessingPipeline(AbstractPreprocessing[T, Any], Generic[T]):
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

    transform_composition: Callable[[T], Any]

    def __init__(
        self, preprocessors: Iterable[AbstractPreprocessing[Any, Any]]
    ) -> None:
        self.preprocessors = preprocessors
        self.is_fitted = False

    def attach_transform_to_dataset(
        self, dataset: Dataset[T]
    ) -> TransformingDataset[T, Any]:
        return TransformingDataset(dataset, self.transform_composition)  # type: ignore

    def fit_to_dataset(self, dataset: Dataset[Any]) -> None:
        self.transform_composition = lambda x: x  # type: ignore
        transformed_dataset = TransformingDataset(dataset, self.transform_composition)  # type: ignore
        for preprocessor in self.preprocessors:
            preprocessor.fit_to_dataset(transformed_dataset)
            transformed_dataset = append_transform(transformed_dataset, preprocessor)
        self.transform_composition = transformed_dataset.transform  # type: ignore

    def __call__(self, x: T) -> Any:
        if not self.is_fitted:
            raise ValueError(
                "The preprocessing pipeline has not been fitted to a dataset."
                " Please call the ´fit_to_dataset´ method first."
            )
        return self.transform_composition(x)  # type: ignore
