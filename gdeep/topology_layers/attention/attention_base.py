from abc import ABC, abstractmethod
from typing import Optional

import torch
from gdeep.topology_layers.persformer_config import PersformerConfig
from torch.nn import Module

# Type aliases
Tensor = torch.Tensor

class AttentionBase(Module, ABC):
    def __init__(self, config: PersformerConfig):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self,  # type: ignore
                input: Tensor,
                attention_mask: Optional[Tensor] = None
                ) -> Tensor:
        raise NotImplementedError
