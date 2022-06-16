
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Dropout, Linear, Module, ModuleList, Sequential

from .attention_factory import AttentionFactory
from .persformer_block import PersformerBlock
from .persformer_config import PersformerConfig
from .utility import get_activation_function, get_pooling_layer

# Type aliases
Tensor = torch.Tensor

class Persformer(Module):
    """Persformer model as described in the paper: https://arxiv.org/abs/2112.15210

    Examples:
    ```python
    >>> from gdeep.topological_layers import PersformerConfig, PersformerModel
    
    # Initialize the configuration object
    >>> config = PersformerConfig()
    
    # Initialize the model
    >>> model = Persformer(config)
    """
    

    
    config: PersformerConfig
    embedding_layer: Module
    persformer_blocks: ModuleList
    classifier_layer: Module
    
    def __init__(self, config: PersformerConfig):
        super().__init__()
        self.config = config
        self.build_model()
        
    def build_model(self):
        """
        Build the model.
        """
        self.embedding_layer = self._get_embedding_layer()
        self.persformer_blocks = ModuleList([self._get_persformer_block() 
                                             for _ in range(self.config.num_attention_layers)])
        self.pooling_layer = self._get_pooling_layer()
        self.classifier_layer = self._get_classifier_layer()

    def _get_embedding_layer(self) -> Module:
        return Sequential(
                            Linear(self.config.input_size, self.config.hidden_size),
                            get_activation_function(self.config.hidden_act),
                        )
        
    def _get_classifier_layer(self) -> Module:
        return Sequential(
                            Linear(self.config.hidden_size, self.config.hidden_size),
                            get_activation_function(self.config.hidden_act),
                            Dropout(self.config.classifier_dropout_prob),
                            Linear(self.config.hidden_size, self.config.output_size),
                        )
        
    def _get_persformer_block(self) -> Module:
        return PersformerBlock(self.config)  # type: ignore
                               
    def _get_pooling_layer(self) -> Module:
        return get_pooling_layer(self.config)
                        
        
    def forward(self,
                input_batch: Tensor,
                attention_mask: Optional[Tensor] = None
                ) -> Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_batch: The input batch. Of shape (batch_size, sequence_length, 2 + num_homology_types)
            attention_mask: The attention mask. Of shape (batch_size, sequence_length)
        
        Returns:
            The logits of the model. Of shape (batch_size, sequence_length, 1)
        """
        # Initialize the output tensor
        output = input_batch
        # Apply the embedding layer
        output = self.embedding_layer(output)
        # Apply the attention layers
        for persformer_block in self.persformer_blocks:
            output = persformer_block(output, attention_mask)
        output = self.pooling_layer(output, attention_mask)
        # Apply the classifier layer
        output = self.classifier_layer(output)
        return output

