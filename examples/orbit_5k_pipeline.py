# %%
from dataclasses import dataclass
import os
from typing import Tuple

import torch
import torch.nn as nn
from gdeep.data import PreprocessingPipeline
from gdeep.data.datasets import PersistenceDiagramFromFiles
from gdeep.data.datasets.base_dataloaders import (DataLoaderBuilder,
                                                  DataLoaderParamsTuples)
from gdeep.data.datasets.persistence_diagrams_from_graphs_builder import \
    PersistenceDiagramFromGraphBuilder
from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import (
    OneHotEncodedPersistenceDiagram, collate_fn_persistence_diagrams)
from gdeep.data.preprocessors import (
    FilterPersistenceDiagramByHomologyDimension,
    FilterPersistenceDiagramByLifetime, NormalizationPersistenceDiagram)
from gdeep.search.hpo import GiottoSummaryWriter
from gdeep.topology_layers import Persformer, PersformerConfig, PersformerWrapper
from gdeep.topology_layers.persformer_config import PoolerType
from gdeep.trainer.trainer import Trainer
from gdeep.search import HyperParameterOptimization
from gdeep.utility import DEFAULT_GRAPH_DIR, PoolerType
from gdeep.utility.utils import autoreload_if_notebook
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import Subset
from torch.utils.tensorboard.writer import SummaryWriter
from gdeep.data.datasets import OrbitsGenerator, DataLoaderKwargs

autoreload_if_notebook()

# %%
@dataclass
class Orbit5kConfig():
    batch_size_train: int = 4
    num_orbits_per_class: int = 32
    validation_percentage: float = 0.0
    test_percentage: float = 0.0
    num_jobs: int = 8
    dynamical_system: str = "classical_convention"
    homology_dimensions: Tuple[int] = (0, 1)  # type: ignore
    dtype: str = "float32"
    arbitrary_precision: bool = False

config_data = Orbit5kConfig()
    

og = OrbitsGenerator(
    num_orbits_per_class=config_data.num_orbits_per_class,
    homology_dimensions=config_data.homology_dimensions,
    validation_percentage=config_data.validation_percentage,
    test_percentage=config_data.test_percentage,
    n_jobs=config_data.num_jobs,
    dynamical_system=config_data.dynamical_system,
    dtype=config_data.dtype,
)


# Define the data loader

dataloaders_dicts = DataLoaderKwargs(
    train_kwargs={"batch_size": config_data.batch_size_train,},
    val_kwargs={"batch_size": 4},
    test_kwargs={"batch_size": 3},
)

if len(config_data.homology_dimensions) == 0:
    dl_train, _, _ = og.get_dataloader_orbits(dataloaders_dicts)
else:
    dl_train, _, _ = og.get_dataloader_persistence_diagrams(dataloaders_dicts)
    
model_config = PersformerConfig(
    input_size=2 + 2, # there are 2 coordinates and 2 homology dimensions
    ouptut_size=5,  # there are 5 classes
    hidden_size=64,
    intermediate_size=128,
    num_attention_layers=2,
    num_attention_heads=8,
)

# %%

# model = Persformer(model_config)

# writer = SummaryWriter()

# loss_function =  nn.CrossEntropyLoss()

# trainer = Trainer(model, [dl_train], loss_function, writer)

# trainer.train(Adam, 3, False, 
#               {"lr":0.01}, 
#               {"batch_size":16})
    
    
# %%
# Define the model by using a Wrapper for the Persformer model

wrapped_model = PersformerWrapper(
    num_attention_layers=3,
    num_attention_heads=4,
    input_size= 2 + 2,
    ouptut_size=5,
    pooler_type=PoolerType.ATTENTION,
)
writer = GiottoSummaryWriter()

loss_function =  nn.CrossEntropyLoss()

trainer = Trainer(wrapped_model, [dl_train, dl_train], loss_function, writer)  # type: ignore

# initialise hpo object
search = HyperParameterOptimization(trainer, "accuracy", 2, best_not_last=True)

# if you want to store pickle files of the models instead of the state_dicts
search.store_pickle = True

# dictionaries of hyperparameters
optimizers_params = {"lr": [0.001, 0.01]}
dataloaders_params = {"batch_size": [2, 4, 2]}
models_hyperparams = {
    "input_size": [4],
    "output_size": [5],
    "num_attention_layers": [1, 2, 1],
    "num_attention_heads": [8, 16, 8],
    "hidden_size": [16],
    "intermediate_size": [16],
}

# %%
# starting the HPO
search.start(
    [Adam],
    3,
    False,
    optimizers_params,
    dataloaders_params,
    models_hyperparams,
)

