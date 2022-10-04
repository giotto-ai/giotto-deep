from abc import ABC, abstractmethod
from typing import Optional

import torch
from gdeep.topology_layers.persformer_config import PersformerConfig
from torch.nn import Module

# Type aliases
from gdeep.utility.custom_types import Tensor


class AttentionBase(Module, ABC):
    """Base class for attention layers. This class
    can be used in generic transformer models.
    """

    def __init__(self, config: PersformerConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self, input: Tensor, attention_mask: Optional[Tensor] = None  # type: ignore
    ) -> Tensor:
        raise NotImplementedError
