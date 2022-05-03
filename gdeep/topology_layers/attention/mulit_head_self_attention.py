from dataclasses import dataclass
from enum import Enum
from typing import Union, TypeVar, List, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

from gdeep.topology_layers.components import build_activation
from gdeep.topology_layers.components import LayerNormStyle

Tensor = torch.Tensor

class AttentionType(Enum):
    """
    A class to define attention types.
    """
    SELF_ATTENTION = "self_attention"
    SPARSE_ATTENTION = "sparse_attention"
    

@dataclass
class MultiHeadAttentionConfig:
    """
    Configuration class to define a multi-head attention layer.
    
    Args:
        dim_model: The dimension of the model.
        num_heads: The number of heads.
        activation: The activation function to use.
        dropout: The dropout probability.
        layer_norm: Whether to use layer normalization.
    """
    dim_model: int
    num_heads: int
    activation: str
    dropout: float
    layer_norm_style: LayerNormStyle
    bias: bool
    attention_type: AttentionType
    
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        layer_norm_style: LayerNormStyle = LayerNormStyle.NONE,
        bias: bool = True,
        attention_type: AttentionType = AttentionType.SELF_ATTENTION,
        ) -> None:
        
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout
        self.layer_norm_style = layer_norm_style
        self.bias = bias
        self.attention_type = attention_type
        
        
class MultiHeadAttentionSelfAttentionLayer(nn.Module):
    """
    A multi-head attention layer.
    
    Args:
        dim_model: The dimension of the model.
        num_heads: The number of heads.
        activation: The activation function to use.
        dropout: The dropout probability.
        layer_norm: Whether to use layer normalization.
    """
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
        ) -> None:
        
        super().__init__()
        
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout
        self.bias = bias
        
        self.multi_head_attention = MultiheadAttention(
            num_heads=num_heads,
            embed_dim=dim_model,
            dropout=dropout,
            bias=bias,
            )
        
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        ) -> Tensor:
        """
        Forward propagation of the multi-head attention layer.
        
        Args:
            query: The query tensor.
            key: The key tensor.
            value: The value tensor.
            mask: The mask tensor.
        
        Returns:
            The multi-head attention output tensor.
        """
        output = self.multi_head_attention(
            query,
            key,
            value,
            mask=mask,
            )
        
        return output