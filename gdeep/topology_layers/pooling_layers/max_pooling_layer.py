from typing import Optional

import torch
from torch.nn import Module

from ..persformer_config import PersformerConfig

# Type aliases
Tensor = torch.Tensor


class MaxPoolingLayer(Module):
    """Implementation of the max pooling layer"""
    config: PersformerConfig
    
    def __init__(self, config: PersformerConfig):
        super().__init__()
        self.config = config
        
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
            The logits of the model. Of shape (batch_size, sequence_length, 1)
        """
        # Initialize the output tensor
        if attention_mask is not None:
            output = input_batch * attention_mask.unsqueeze(2)
        else:
            output = input_batch 
        # Apply the max pooling layer
        output = output.max(dim=-2)[0]
        return output
