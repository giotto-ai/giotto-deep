from typing import Tuple
from dataclasses import dataclass
import argparse
import functools

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from gdeep.search.hpo import GiottoSummaryWriter
from gdeep.topology_layers import PersformerWrapper, persformer_block
from gdeep.topology_layers.persformer_config import PoolerType
from gdeep.trainer.trainer import Trainer, Parallelism
from gdeep.data.datasets import OrbitsGenerator, DataLoaderKwargs
import gdeep.utility_examples.args

def main(args):
    # Generate a configuration file with the parameters of the desired dataset
    @dataclass
    class Orbit5kConfig():
        batch_size_train: int = args.batch_size
        num_orbits_per_class: int = 1000
        validation_percentage: float = 0.0
        test_percentage: float = 0.0
        num_jobs: int = 8
        dynamical_system: str = "classical_convention"
        homology_dimensions: Tuple[int, int] = (0, 1)
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

    # Define the model by using a Wrapper for the Persformer model

    if args.big_model:
        # Big model
        wrapped_model = PersformerWrapper(
            num_attention_layers=8,
            num_attention_heads=32,
            input_size= 2 + 2,
            output_size=5,
            pooler_type=PoolerType.ATTENTION,
            hidden_size=128,
            intermediate_size=128,
        )
        config_mha = [{'embed_dim': 128, 'num_heads': 32, 'dropout': 0.1, 'batch_first': True}] * 9
    else:
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
        config_mha = [{'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True}] * 5

    # Define the trainer

    writer = GiottoSummaryWriter()
    loss_function = nn.CrossEntropyLoss()
    trainer = Trainer(wrapped_model, [dl_train, dl_train], loss_function, writer)
    devices = list(range(torch.cuda.device_count()))
    config_fsdp = {
        "sharding_strategy": args.sharding.to_sharding_strategy(),
        "auto_wrap_policy": functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={persformer_block.PersformerBlock,}),
        }
    parallel = Parallelism(args.parallel,
                           devices,
                           len(devices),
                           config_fsdp=config_fsdp,
                           config_mha=config_mha,
                           pipeline_chunks=2)

    # train the model

    return trainer.train(Adam, args.n_epochs, parallel=parallel)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Orbit 5k example")
    gdeep.utility_examples.args.add_default_arguments(parser)
    gdeep.utility_examples.args.add_big_model(parser)
    args = parser.parse_args()
    main(args)
