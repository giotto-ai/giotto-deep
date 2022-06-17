from typing import Optional

import torch
from gdeep.utility.enum_types import LayerNormStyle
from torch.nn import LayerNorm, Module, ModuleList

from .persformer_config import PersformerConfig
from .utility import get_attention_layer, get_feed_forward_layer

# Type aliases
Tensor = torch.Tensor

class PersformerBlock(Module):
    """
    A persformer block.
    """
    
    config: PersformerConfig
    attention_layer: Module
    feed_forward_layer: Module
    dropout_layer: Module
    layer_norms: Optional[ModuleList]
    
    def __init__(self, config: PersformerConfig):
        super().__init__()
        self.config = config
        self.build_model()
        
    def build_model(self):
        """
        Build the model.
        """
        self.attention_layer = get_attention_layer(self.config)
        if not self.config.use_attention_only:
            self.feed_forward_layer = get_feed_forward_layer(self.config)
       
        if self.config.layer_norm_style == LayerNormStyle.PRE_LAYER_NORMALIZATION or \
            self.config.layer_norm_style == LayerNormStyle.POST_LAYER_NORMALIZATION:
            self.layer_norms = ModuleList([LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps) 
                                           for _ in range(2)])
        
    def forward(self,  # type: ignore
                input_batch: Tensor,
                attention_mask: Optional[Tensor] = None
                ) -> Tensor:
        """
        Forward pass of the model. We implement different layer normalizations in the forward pass,
        see the paper https://arxiv.org/pdf/2002.04745.pdf for details.
        
        Args:
            input_batch:
                The input batch. Of shape (batch_size, sequence_length, 2 + num_homology_types)
            attention_mask:
                The attention mask. Of shape (batch_size, sequence_length)
        
        Returns:
            The logits of the model. Of shape (batch_size, sequence_length, 1)
        """
        if self.config.layer_norm_style == LayerNormStyle.NO_LAYER_NORMALIZATION:
            return self._forward_no_layer_norm(input_batch, attention_mask)
        elif self.config.layer_norm_style == LayerNormStyle.PRE_LAYER_NORMALIZATION:
            return self._forward_pre_layer_norm(input_batch, attention_mask)
        elif self.config.layer_norm_style == LayerNormStyle.POST_LAYER_NORMALIZATION:
            return self._forward_post_layer_norm(input_batch, attention_mask)
        else:
            raise ValueError(f"Unknown layer norm style {self.config.layer_norm_style}")
        
        
    def _forward_no_layer_norm(self,
                               input_batch: Tensor,
                               attention_mask: Optional[Tensor] = None
                               ) -> Tensor:
        """
        Forward pass of the model without layer normalization.
        
        Args:
            input_batch:
                The input batch. Of shape (batch_size, sequence_length, 2 + num_homology_types)
            attention_mask:
                The attention mask. Of shape (batch_size, sequence_length)
        
        Returns:
            The logits of the model. Of shape (batch_size, sequence_length, 1)
        """
        output = self.attention_layer(input_batch, attention_mask) + input_batch  # type: ignore
        if not self.config.use_attention_only:
            output = self.feed_forward_layer(output) + output
        return output
        
    def _forward_pre_layer_norm(self,
                                input_batch: Tensor,
                                attention_mask: Optional[Tensor] = None
                                ) -> Tensor:
        """
        Forward pass of the model with pre-layer normalization.
        
        Args:
            input_batch:
                The input batch. Of shape (batch_size, sequence_length, 2 + num_homology_types)
            attention_mask:
                The attention mask. Of shape (batch_size, sequence_length)
        
        Returns:
            The logits of the model. Of shape (batch_size, sequence_length, 1)
        """
        assert self.layer_norms is not None
        normalized = self.layer_norms[0](input_batch)
        output = self.attention_layer(normalized, attention_mask) + input_batch
        if not self.config.use_attention_only:
            normalized = self.layer_norms[1](output)
            output = self.feed_forward_layer(normalized) + output
        return output
    
    def _forward_post_layer_norm(self,
                                 input_batch: Tensor,
                                 attention_mask: Optional[Tensor] = None
                                 ) -> Tensor:
        """
        Forward pass of the model with post-layer normalization.
        
        Args:
            input_batch:
                The input batch. Of shape (batch_size, sequence_length, 2 + num_homology_types)
            attention_mask:
                The attention mask. Of shape (batch_size, sequence_length)
        
        Returns:
            The logits of the model. Of shape (batch_size, sequence_length, 1)
        """
        assert self.layer_norms is not None
        output = self.attention_layer(input_batch, attention_mask) + input_batch
        output = self.layer_norms[0](output)
        if not self.config.use_attention_only:
            output = self.feed_forward_layer(output) + output
            output = self.layer_norms[1](output)
        return output
