from sklearn import preprocessing
from gdeep.data import Normalize, \
    PreprocessImageClassification, BasicDataset, \
    DatasetImageClassificationFromFiles
import os
from ..preprocessing_pipeline import PreprocessingPipeline
from ..transforming_dataset import TransformingDataset


def test_preeprocessing_pipeline():
    """test preprocessing pipeline"""
    normalize = Normalize()
    file_path = os.path.dirname(os.path.realpath(__file__))

    image_dataset = DatasetImageClassificationFromFiles(
        os.path.join(file_path, "img_data"),
        os.path.join(file_path, "img_data", "labels.csv"))
    # define preprocessing pipeline
    preprocessing_pipeline = PreprocessingPipeline((normalize,))

    # fit preprocessing pipeline to dataset
    preprocessing_pipeline.fit_to_dataset(image_dataset)
    
    # apply preprocessing pipeline to dataset
    preprocessed_dataset = TransformingDataset(image_dataset,
                                                  preprocessing_pipeline.transform)
    
    # TODO: check if preprocessing pipeline is applied correctly
    
