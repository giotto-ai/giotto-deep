from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional
from transformers.configuration_utils import PretrainedConfig
import torch
import torch.nn as nn
from torch.nn import Module, MultiheadAttention, Linear, Sequential, LayerNorm, Dropout, ModuleList

from gdeep.utility.enum_types import ActivationFunction, PoolerType, AttentionType, LayerNormStyle

# Type aliases
Tensor = torch.Tensor
    

class PersformerConfig(PretrainedConfig):
    """
    Configuration class to define a persformer model.
    
    Examples:
    ```python
    >>> from gdeep.topological_layers import PersformerConfig, PersformerModel
    
    # Initialize the configuration object
    >>> config = PersformerConfig()
    
    # Initialize the model
    >>> model = Persformer(config)
    
    # Access the configuration object
    >>> config = model.config
    
    ```
    """
    
    input_size: int # input size of the model
    output_size: int
    hidden_size: int
    num_attention_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_act: ActivationFunction
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    layer_norm_eps: float
    classifier_dropout_prob: float
    layer_norm_style: LayerNormStyle
    attention_type: AttentionType
    activation_fn: ActivationFunction
    pooler_type: PoolerType
    use_attention_only: bool
    use_skip_connections_for_persformer_blocks: bool
         
    
    def __init__(self,
                 input_size: int = 2 + 4,
                 output_size: int = 2,
                 hidden_size: int = 32,
                 num_attention_layers: int = 2,
                 num_attention_heads: int = 4,
                 intermediate_size: int = 32,
                 hidden_act: ActivationFunction = ActivationFunction.GELU,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 layer_norm_eps: float = 1e-12,
                 classifier_dropout_prob: float = 0.1,
                 use_layer_norm: LayerNormStyle = \
                     LayerNormStyle.NO_LAYER_NORMALIZATION,
                 attention_type: AttentionType = \
                     AttentionType.DOT_PRODUCT,
                 pooler_type: PoolerType = PoolerType.ATTENTION,
                 use_attention_only: bool = False,
                 use_skip_connections_for_persformer_blocks=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)  # type: ignore
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout_prob = classifier_dropout_prob
        self.layer_norm_style = use_layer_norm
        self.attention_type = attention_type
        self.pooler_type = pooler_type
        self.use_attention_only = use_attention_only
        self.use_skip_connections_for_persformer_blocks = use_skip_connections_for_persformer_blocks


