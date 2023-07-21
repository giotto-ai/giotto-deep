# Include necessary general imports
import os
from typing import Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import argparse

# Torch imports

import torch
import torch.nn as nn

# Gdeep imports 

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
from gdeep.topology_layers import Persformer, PersformerConfig, PersformerWrapper, persformer_block
from gdeep.topology_layers.persformer_config import PoolerType
from gdeep.trainer.trainer import Trainer, Parallelism, ParallelismType
from gdeep.search import HyperParameterOptimization
from gdeep.utility import DEFAULT_GRAPH_DIR, PoolerType
from gdeep.utility.utils import autoreload_if_notebook
from gdeep.analysis.interpretability import Interpreter
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import Subset
from gdeep.visualization import Visualiser
from gdeep.data.datasets import OrbitsGenerator, DataLoaderKwargs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Giotto FSDP Example')
    parser.add_argument('--fsdp', action='store_true', default=False,
                            help='Enable FSDP for training')
    parser.add_argument('--layer_cls', action='store_true', default=False,
                            help='Use layer class')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--n_epochs', type=int, default=1, metavar='N',
                        help='Number of epochs to train for (default: 1)')
    args = parser.parse_args()

    # Generate a configuration file with the parameters of the desired dataset
    @dataclass
    class Orbit5kConfig():
        batch_size_train: int = args.batch_size
        num_orbits_per_class: int = 1000
        validation_percentage: float = 0.0
        test_percentage: float = 0.0
        num_jobs: int = 8
        dynamical_system: str = "classical_convention"
        homology_dimensions: Tuple[int, int] = (0, 1)  # type: ignore
        dtype: str = "float32"
        arbitrary_precision: bool = False

    config_data = Orbit5kConfig()

    # Define the OrbitsGenerator Class with the above parameters    

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
        
    # Get the orbits point clouds

    point_clouds = og.get_orbits()

    # For each rho value, plot one point cloud

    rho_values = [2.5, 3.5, 4.0, 4.1, 4.3]
    #fig, ax = plt.subplots(ncols=len(rho_values), figsize = (20,3))

    #for i in range(len(rho_values)):
    #    x , y = point_clouds[i*config_data.num_orbits_per_class,:,0], point_clouds[i*config_data.num_orbits_per_class,:,1] 
    #    ax[i].scatter(x, y)
    #    ax[i].set_title('Example of orbit for rho = ' + str(rho_values[i]))

    # Define the model by using a Wrapper for the Persformer model

    # Big model
    # wrapped_model = PersformerWrapper(
    #     num_attention_layers=8,
    #     num_attention_heads=32,
    #     input_size= 2 + 2,
    #     output_size=5,
    #     pooler_type=PoolerType.ATTENTION,
    #     hidden_size=128,
    #     intermediate_size=128,
    # )

    # Small model
    wrapped_model = PersformerWrapper(
        num_attention_layers=2,
        num_attention_heads=8,
        input_size= 2 + 2,
        output_size=5,
        pooler_type=PoolerType.ATTENTION,
        hidden_size=16,
        intermediate_size=16,
    )

    # Define the trainer 

    writer = GiottoSummaryWriter()

    loss_function =  nn.CrossEntropyLoss()

    trainer = Trainer(wrapped_model, [dl_train, dl_train], loss_function, writer) 

    devices = list(range(torch.cuda.device_count()))

    parallel = Parallelism(ParallelismType.FSDP_ZERO2, 
                           devices, 
                           len(devices), 
                           transformer_layer_class=persformer_block.PersformerBlock if args.layer_cls else None)

    # train the model for one epoch

    n_epoch = args.n_epochs

    trainer.train(Adam, n_epoch, parallel=parallel if args.fsdp else None)

    exit()

    # Initialize the Interpreter class in Saliency mode

    inter = Interpreter(trainer.model, method="Saliency")

    # Get a datum and its corresponding class

    batch = next(iter(dl_train))
    datum = batch[0][0].reshape(1, *(batch[0][0].shape))
    class_ = batch[1][0].item()

    # interpret the diagram
    x, attr = inter.interpret(x=datum, y=class_)

    # visualise the results
    vs = Visualiser(trainer)
    vs.plot_attributions_persistence_diagrams(inter)
