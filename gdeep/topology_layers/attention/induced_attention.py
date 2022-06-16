from typing import Optional

import torch

from ..persformer_config import PersformerConfig
from .attention_base import AttentionBase

# Type aliases
Tensor = torch.Tensor


class InducedAttention(AttentionBase):
    def __init__(self, config: PersformerConfig) -> None:
        super().__init__(config)
        raise NotImplementedError
    
    def forward(self,  # type: ignore
                input: Tensor,
                attention_mask: Optional[Tensor] = None
                ):
        raise NotImplementedError
    