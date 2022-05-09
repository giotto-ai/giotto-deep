from enum import Enum

import torch
import torch.nn as nn

def build_activation(activation: str) -> nn.Module:
    """
    Build an activation function.
    
    Args:
        activation: The activation function to use.
        
    Returns:
        activation: The activation function.
        
    Raises:	
        ValueError: If the activation function is not supported.
        
    Examples:
        >>> activation = build_activation("gelu")
        >>> activation
        <torch.nn.modules.activation.GELU>
    """
    if activation == "gelu":
        return nn.GELU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "swish":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {activation}")
    
    
class LayerNormStyle(Enum):
    """
    A class to define layer normalization styles.
    """
    POST = "post"
    PRE = "pre"
    NONE = "none"
        
