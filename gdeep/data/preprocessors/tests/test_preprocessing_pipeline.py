from sklearn import preprocessing
from gdeep.data.datasets import ImageClassificationFromFiles
from gdeep.data.preprocessors import ToTensorImage
import os
from ...preprocessing_pipeline import PreprocessingPipeline
from gdeep.data.preprocessors import Normalization
from ...transforming_dataset import TransformingDataset


def test_preeprocessing_pipeline():
    """test preprocessing pipeline"""
    normalize = Normalization()
    to_tensor = ToTensorImage([32,32])
    file_path = os.path.dirname(os.path.realpath(__file__))

    image_dataset = ImageClassificationFromFiles(
        os.path.join(file_path, "img_data"),
        os.path.join(file_path, "img_data", "labels.csv"))
    # define preprocessing pipeline
    preprocessing_pipeline = PreprocessingPipeline((to_tensor, normalize))

    # fit preprocessing pipeline to dataset
    preprocessing_pipeline.fit_to_dataset(image_dataset)
    
    # apply preprocessing pipeline to dataset
    preprocessed_dataset = preprocessing_pipeline.attach_transform_to_dataset(image_dataset)
    
    assert len(preprocessed_dataset[0][0].shape) == 3
    
