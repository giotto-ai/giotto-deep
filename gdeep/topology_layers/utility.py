import torch
import torch.nn as nn
from gdeep.topology_layers.pooling_layers import (AttentionPoolingLayer,
                                                  MaxPoolingLayer,
                                                  MeanPoolingLayer,
                                                  SumPoolingLayer)
from gdeep.utility.enum_types import ActivationFunction, PoolerType
from torch.nn import Dropout, Linear, Module, Sequential

from .attention_factory import AttentionFactory
from .persformer_config import PersformerConfig


def get_pooling_layer(config: PersformerConfig) -> Module:
    """
    Get the pooling layer.
    
    Args:
        config: The configuration of the model.
        
    Returns:
        The pooling layer.
    """
    if(config.pooler_type is PoolerType.ATTENTION):
        return AttentionPoolingLayer(config)  # type: ignore
    elif(config.pooler_type is PoolerType.MAX):
        return MaxPoolingLayer(config)  # type: ignore
    elif(config.pooler_type is PoolerType.MEAN):
        return MeanPoolingLayer(config)  # type: ignore
    elif(config.pooler_type is PoolerType.SUM):
        return SumPoolingLayer(config)  # type: ignore
    else:
        raise ValueError(f"Pooler type {config.pooler_type} is not supported.")



def get_feed_forward_layer(config: PersformerConfig) -> Module:
    """
    Get the feed forward layer.
    """
    feed_forward_layer = Sequential()
    feed_forward_layer.add_module(
                                    "intermediate",
                                    Linear(config.hidden_size, config.intermediate_size)
                                )
    feed_forward_layer.add_module(
                                    "activation",
                                    get_activation_function(config.hidden_act)
                                )
    feed_forward_layer.add_module(
                                    "dropout",
                                    Dropout(config.hidden_dropout_prob)
                                )
    
    feed_forward_layer.add_module(
                                    "output",
                                    Linear(config.intermediate_size, config.hidden_size)
                                    )
    feed_forward_layer.add_module(
                                    "dropout",
                                    Dropout(config.hidden_dropout_prob)
                                )
    return feed_forward_layer


def get_attention_layer(config: PersformerConfig) -> Module:
    """
    Get the attention layer.
    """
    return AttentionFactory().build(config)  # type: ignore


def get_activation_function(activation_function: ActivationFunction) -> Module:
    """
    Get the activation function.
    """
    if(activation_function is ActivationFunction.RELU):
        return nn.ReLU()
    elif(activation_function is ActivationFunction.GELU):
        return nn.GELU()
    elif(activation_function is ActivationFunction.SELU):
        return nn.SELU()
    elif(activation_function is ActivationFunction.MISH):
        return nn.Mish()
    else:
        raise ValueError("Unknown activation function.")
