from typing import Optional

import torch
from torch.nn import Module

from ..persformer_config import PersformerConfig

# Type aliases
Tensor = torch.Tensor

class MeanPoolingLayer(Module):
        
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
                input_batch: The input batch. Of shape (batch_size, sequence_length, hidden_size)
                
            Returns:
                The pooled output. Of shape (batch_size, hidden_size)
            """
            # Initialize the output tensor
            if attention_mask is not None:
                output = input_batch * attention_mask.unsqueeze(dim=-1)
                output = output.sum(dim=-2) / attention_mask.sum(dim=-1, keepdim=True)
                return output
            else:
                output = input_batch
                return output.mean(dim=-2)