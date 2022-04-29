from dataclasses import dataclass
from typing import Union, TypeVar, List, Optional, Dict, Any

import torch
import torch.nn as nn

from gdeep.topology_layers.components import build_activation

# Define Type Self
Self = TypeVar("Self", bound="Feedforward")
 

@dataclass
class FeedforwardConfig:
    """
    Configuration class to define a feedforward layer.
    
    Args:
        dim_model: The dimension of the model.
        dim_feedforward: The dimension of the feedforward layer.
        activation: The activation function to use.
        dropout: The dropout probability.
        layer_norm: Whether to use layer normalization.
    """
    dim_model: int
    dim_feedforward: int
    activation: str
    dropout: float
    layer_norm: str
    layer_factor: float
    bias: bool
    
    def __init__(
        self,
        dim_model: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        layer_norm: str = "post",
        layer_factor: float = 1.0,
        bias: bool = True,
        ) -> None:
        
        self.dim_model = dim_model
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.layer_factor = layer_factor
        self.bias = bias
        
class Feedforward(nn.Module):
    """
    A feedforward layer.
    
    Args:
        dim_model: The dimension of the model.
        dim_feedforward: The dimension of the feedforward layer.
        activation: The activation function to use.
        dropout: The dropout probability.
        layer_norm: Whether to use layer normalization.
    """
    def __init__(
        self,
        dim_model: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        layer_factor: float = 1.0,
        bias = True,
        ) -> None:
        
        super().__init__()
        
        
        self.activation_layer = build_activation(activation)
            
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, dim_model * layer_factor, bias=bias),
            nn.Dropout(dropout),
            nn.Linear(dim_model * layer_factor, dim_model, bias=bias),
            nn.Dropout(dropout),
        )
    
    @classmethod
    def from_config(
        cls: Type[Self],
        config: FeedforwardConfig,
        ) -> Self:
        
        return cls(
            config.dim_model,
            config.dim_feedforward,
            config.activation,
            config.dropout,
            config.layer_norm,
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: The input tensor.
            
        Returns:
            x: The output tensor.
        """
        x = self.mlp(x)
        return x