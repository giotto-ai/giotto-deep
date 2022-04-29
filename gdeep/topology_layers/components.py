import torch
import torch.nn as nn

def build_activation(activation: str) -> nn.Module:
    """
    Build an activation function.
    
    Args:
        activation: The activation function to use.
        
    Returns:
        activation: The activation function.
    """
    if activation == "gelu":
        return nn.GELU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "swish":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {activation}")