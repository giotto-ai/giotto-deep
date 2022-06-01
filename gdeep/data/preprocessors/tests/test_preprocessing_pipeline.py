import os
from typing import Tuple

import torch

from gdeep.data.datasets import ImageClassificationFromFiles
from gdeep.data.datasets.persistence_diagrams_from_files import \
    PersistenceDiagramFromFiles
from gdeep.data.datasets.persistence_diagrams_from_graphs_builder import \
    PersistenceDiagramFromGraphBuilder
from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import \
    OneHotEncodedPersistenceDiagram
from gdeep.data.preprocessors import Normalization, ToTensorImage
from gdeep.data.preprocessors.filter_persistence_diagram_by_homology_dimension import \
    FilterPersistenceDiagramByHomologyDimension
from gdeep.data.preprocessors.filter_persistence_diagram_by_lifetime import \
    FilterPersistenceDiagramByLifetime
from gdeep.data.preprocessors.normalization_persistence_diagram import \
    NormalizationPersistenceDiagram
from gdeep.utility.constants import DEFAULT_GRAPH_DIR
from gdeep.data.preprocessors.normalization import _compute_mean_of_dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from ...preprocessing_pipeline import PreprocessingPipeline
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
    

def test_preprocessing_pipeline_for_persistence_diagrams():
    # Parameters
    name_graph_dataset: str = 'MUTAG'
    diffusion_parameter: float = 0.1
    num_homology_types: int = 4


    # Create the persistence diagram dataset
    pd_creator = PersistenceDiagramFromGraphBuilder(name_graph_dataset, diffusion_parameter)
    pd_creator.create()
    
    pd_mutag_ds = PersistenceDiagramFromFiles(
    os.path.join(
        DEFAULT_GRAPH_DIR, f"MUTAG_{diffusion_parameter}_extended_persistence"
        )
    )
    # Create the train/validation/test split

    train_indices, test_indices = train_test_split(
        range(len(pd_mutag_ds)),
        test_size=0.2,
        random_state=42,
    )

    train_indices , validation_indices = train_test_split(
        train_indices,
        test_size=0.2,
        random_state=42,
    )

    # Create the data loaders
    train_dataset = Subset(pd_mutag_ds, train_indices)
    validation_dataset = Subset(pd_mutag_ds, validation_indices)
    test_dataset = Subset(pd_mutag_ds, test_indices)

    # Preprocess the data
    preprocessing_pipeline = PreprocessingPipeline[Tuple[OneHotEncodedPersistenceDiagram, int]](
        (
            FilterPersistenceDiagramByHomologyDimension[int]([0, 1]),
            FilterPersistenceDiagramByLifetime[int](min_lifetime=-0.1, max_lifetime=1.0),
            NormalizationPersistenceDiagram[int](num_homology_dimensions=4),
        )
    )

    preprocessing_pipeline.fit_to_dataset(train_dataset)
    
    train_dataset = preprocessing_pipeline.attach_transform_to_dataset(train_dataset)
    validation_dataset = preprocessing_pipeline.attach_transform_to_dataset(validation_dataset)
    test_dataset = preprocessing_pipeline.attach_transform_to_dataset(test_dataset)

    
    computed_mean = _compute_mean_of_dataset(
                TransformingDataset(train_dataset, lambda x: (x[0].get_raw_data().mean(dim=0), x[1]))
                )[:2]
    # -> (0.0, 0.0, sth, sth)
    assert torch.allclose(computed_mean.float(), torch.tensor([0.0, 0.0]), atol=1e-4)
    computed_stddev= _compute_mean_of_dataset(
                TransformingDataset(train_dataset, lambda x: ((x[0].get_raw_data()**2).mean(dim=0), x[1]))
                )[:2]
    # -> (1.0, 1.0, sth, sth)
    assert torch.allclose(computed_stddev.float(), torch.tensor([1.0, 1.0]), atol=1e-4)
