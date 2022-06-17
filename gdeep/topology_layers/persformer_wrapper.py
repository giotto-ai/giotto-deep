
from typing import Optional

import torch
from gdeep.utility.enum_types import (ActivationFunction, AttentionType,
                                      LayerNormStyle, PoolerType)
from torch.nn import Module

from .persformer import Persformer
from .persformer_config import PersformerConfig

Tensor = torch.Tensor


class PersformerWrapper(Module):
    """The wrapper for persformer to allow compatibility
    with the HPO classes.
    """
    config: PersformerConfig
    model: Module
    
    def __init__(self,
                 **kwargs):
        super().__init__()
        self.config = PersformerConfig(
            **kwargs
        )
        self.model = Persformer(self.config)
    
    def forward(self, 
                input: Tensor,
                attention_mask: Optional[Tensor] = None
                ):
        return self.model(input, attention_mask)
