"""Modify arguments of examples."""

import argparse
from gdeep.trainer.trainer import ParallelismType
from gdeep.utility_examples.fsdp import ShardingStrategyEx


def add_default_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--parallel',
                        type=ParallelismType.from_str,
                        default=ParallelismType._NONE,
                        help='Parallelism type to use for training (default: none)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=4,
                        metavar='N',
                        help='input batch size for training (default: %(default)s)')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=1,
                        metavar='N',
                        help='Number of epochs to train for (default: %(default)s)')
    parser.add_argument('--sharding',
                        type=ShardingStrategyEx.from_str,
                        choices=[x for x in ShardingStrategyEx],
                        default=ShardingStrategyEx.SHARD_GRAD_OP,
                        help='Sharding strategy for FSDP (default: %(default)s)')


def add_big_model(parser: argparse.ArgumentParser):
    parser.add_argument('--big-model',
                        action='store_true',
                        help='Use the big model')
