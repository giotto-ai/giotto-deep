# %%
from dataclasses import dataclass
import os
from shutil import rmtree
from typing import Any, Callable, List, Literal, Tuple
from sklearn.model_selection import train_test_split

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Adam

from gdeep.data.datasets import PersistenceDiagramFromFiles
from gdeep.data.datasets.base_dataloaders import DataLoaderBuilder, DataLoaderParamsTuples
from gdeep.data.datasets.persistence_diagrams_from_graphs_builder import PersistenceDiagramFromGraphBuilder
from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import OneHotEncodedPersistenceDiagram, collate_fn_persistence_diagrams
from gdeep.data import PreprocessingPipeline
from gdeep.data.preprocessors import (
    NormalizationPersistenceDiagram,
    FilterPersistenceDiagramByLifetime,
    FilterPersistenceDiagramByHomologyDimension
)
from gdeep.data.preprocessors.normalization import _compute_mean_of_dataset
from gdeep.data.transforming_dataset import TransformingDataset
from gdeep.topology_layers import PersformerConfig, Persformer
from gdeep.topology_layers.persformer_config import PoolerType
from gdeep.trainer.trainer import Trainer
from gdeep.utility.utils import autoreload_if_notebook
from gdeep.utility import DEFAULT_GRAPH_DIR

autoreload_if_notebook()

# Parameters
name_graph_dataset: str = 'MUTAG'
diffusion_parameter: float = 0.1
num_homology_types: int = 4


# Create the persistence diagram dataset
pd_creator = PersistenceDiagramFromGraphBuilder(name_graph_dataset, diffusion_parameter)
pd_creator.create()
# %%
# Plot sample extended persistence diagram
file_path: str = os.path.join(DEFAULT_GRAPH_DIR,
                              f"MUTAG_{diffusion_parameter}_extended_persistence", "diagrams")
graph_idx = 1
pd: OneHotEncodedPersistenceDiagram = \
    OneHotEncodedPersistenceDiagram.load(os.path.join(file_path, 
                                                      f"{graph_idx}.npy"))
pd.set_homology_dimension_names(["Ord0", "Ext0", "Rel1", "Ext1"])
pd.plot()
# %%

pd_mutag_ds = PersistenceDiagramFromFiles(
    os.path.join(
        DEFAULT_GRAPH_DIR, f"MUTAG_{diffusion_parameter}_extended_persistence"
        )
)

pd: OneHotEncodedPersistenceDiagram = pd_mutag_ds[0][0]

fig = pd.plot(["Ord0", "Ext0", "Rel1", "Ext1"])
# add title
fig.show()
# %%

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

# %%

train_dataset = preprocessing_pipeline.attach_transform_to_dataset(train_dataset)
validation_dataset = preprocessing_pipeline.attach_transform_to_dataset(validation_dataset)
test_dataset = preprocessing_pipeline.attach_transform_to_dataset(test_dataset)

# %%

dl_params = DataLoaderParamsTuples.default(
    batch_size=32,
    num_workers=0,
    collate_fn=collate_fn_persistence_diagrams,
    with_validation=True,
)


# Build the data loaders
dlb = DataLoaderBuilder((train_dataset, validation_dataset, test_dataset))  # type: ignore
dl_train, dl_val, dl_test = dlb.build(dl_params)  # type: ignore
#%%

# Define the model
model_config = PersformerConfig(
    num_layers=6,
    num_heads=8,
    input_size= 2 + num_homology_types,
    pooler_type=PoolerType.ATTENTION,
)

model = Persformer(model_config)
writer = SummaryWriter()

loss_function =  nn.CrossEntropyLoss()

# trainer = Trainer(model, [dl_train, dl_val, dl_test], loss_function, writer)

# trainer.train(Adam, 3, False, {"lr":0.01}, {"batch_size":16, "collate_fn":collate_fn_persistence_diagrams})

# %%
input, mask, labels = next(iter(dl_train))
model.forward(input, mask)

# %%
# %%
def fun(item: Literal["loss", "accuracy"]):
    # check if the item is a SearchMetrics
    if item not in ["loss", "accuracy"]:
        raise ValueError(f"{item} is not a valid search metric")
    print(item)

fun("asf")
# %%
