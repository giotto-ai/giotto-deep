from typing import Optional

import torch
from torch.nn import Dropout, MultiheadAttention

from ..persformer_config import PersformerConfig
from .attention_base import AttentionBase

# Type aliases
Tensor = torch.Tensor


class ScaledDotProductAttention(AttentionBase):
    """
    Dot product attention. See https://arxiv.org/abs/1706.03762.
    """
    def __init__(self,
                 config: PersformerConfig):
        super().__init__(config)
        
        self.scaled_dot_product_attention = \
            MultiheadAttention(embed_dim=config.hidden_size,
                               num_heads=config.num_attention_heads,
                               dropout=config.attention_probs_dropout_prob,
                               batch_first=True)
        self.dropout = Dropout(config.hidden_dropout_prob)
    
    
    def forward(self,  # type: ignore
                input: Tensor,
                attention_mask: Optional[Tensor] = None
                ):
        """
        Forward pass.
        """
        # if attention_mask is not None:
        #     attention_mask = torch.stack([attention_mask] * attention_mask.shape[-1], dim=-1)
        #     attn_mask = torch.concat([attention_mask] * self.config.num_attention_heads, dim=0)
        # else:
        #     attn_mask = None
        attention_output, _ = self.scaled_dot_product_attention(input, input, input,
                                                                key_padding_mask=attention_mask)
        return self.dropout(attention_output)
