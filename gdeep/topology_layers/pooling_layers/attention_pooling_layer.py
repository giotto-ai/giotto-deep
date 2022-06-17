from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Module, MultiheadAttention

from ..persformer_config import PersformerConfig

# Type aliases
Tensor = torch.Tensor


class AttentionPoolingLayer(Module):
    """This class implements the attention mechanism
    with a pooling layer to enforce permutation
    invariance"""
    config: PersformerConfig
    
    def __init__(self, config: PersformerConfig):
        super().__init__()
        self.config = config
        self.query = nn.parameter.Parameter(torch.Tensor(1, config.hidden_size),
                                              requires_grad=True)
        self.scaled_dot_product_attention = \
            MultiheadAttention(embed_dim=config.hidden_size,
                               num_heads=config.num_attention_heads,
                               dropout=config.attention_probs_dropout_prob,
                               batch_first=True)

        
    def forward(self,
                input_batch: Tensor,
                attention_mask: Optional[Tensor] = None
                ) -> Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_batch:
                The input batch. Of shape (batch_size, sequence_length, hidden_size)
            
        Returns:
            The pooled output. Of shape (batch_size, hidden_size)
        """
        output, _ = self.scaled_dot_product_attention(
                                                 self.query.expand(input_batch.shape[0], -1, -1),
                                                 input_batch,
                                                 input_batch,
                                                 key_padding_mask=attention_mask
                                                 )
        return output.squeeze(dim=1)
