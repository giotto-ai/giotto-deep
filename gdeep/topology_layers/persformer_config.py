from enum import Enum, auto
from typing import Callable, Dict, Optional
from transformers.configuration_utils import PretrainedConfig
import torch
import torch.nn as nn
from torch.nn import Module, MultiheadAttention, Linear, Sequential, LayerNorm, Dropout, ModuleList

# Type aliases
Tensor = torch.Tensor

class LayerNormStyle(Enum):
    """
    The style of layer normalization.
    """
    NO_LAYER_NORMALIZATION = auto()
    PRE_LAYER_NORMALIZATION = auto()
    POST_LAYER_NORMALIZATION = auto()
    
class AttentionType(Enum):
    """
    The type of attention.
    """
    NO_ATTENTION = auto()
    DOT_PRODUCT = auto()
    UNNORMALIZED_DOT_PRODUCT = auto()
    INDUCED_ATTENTION = auto()
    FOURIER_MIXER = auto()

class ActivationFunction(Enum):
    """
    The activation function.
    """
    RELU = auto()
    GELU = auto()
    SELU = auto()
    MISH = auto()

def get_activation_function(activation_function: ActivationFunction) -> Module:
    """
    Get the activation function.
    """
    if(activation_function == ActivationFunction.RELU):
        return nn.ReLU()
    elif(activation_function == ActivationFunction.GELU):
        return nn.GELU()
    elif(activation_function == ActivationFunction.SELU):
        return nn.SELU()
    elif(activation_function == ActivationFunction.MISH):
        return nn.Mish()
    else:
        raise ValueError("Unknown activation function.")
    

class PersformerConfig(PretrainedConfig):
    """
    Configuration class to define a persformer model.
    
    Examples:
    ```python
    >>> from gdeep.topological_layers import PersformerConfig, PersformerModel
    
    # Initialize the configuration object
    >>> config = PersformerConfig()
    
    # Initialize the model
    >>> model = PersformerModel(config)
    
    # Access the configuration object
    >>> config = model.config
    
    ```
    """
    
    input_size: int # input size of the model
    output_size: int
    hidden_size: int
    num_attention_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_act: ActivationFunction
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    layer_norm_eps: float
    classifier_dropout_prob: float
    layer_norm_style: LayerNormStyle
    attention_type: AttentionType
    activation_fn: ActivationFunction
         
    
    def __init__(self,
                 input_size: int = 4,
                 ouptut_size: int = 2 + 4,
                 hidden_size: int = 32,
                 num_attention_layers: int = 2,
                 num_attention_heads: int = 4,
                 intermediate_size: int = 32,
                 hidden_act: ActivationFunction = ActivationFunction.GELU,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 layer_norm_eps: float = 1e-12,
                 classifier_dropout_prob: float = 0.1,
                 use_layer_norm: LayerNormStyle = \
                     LayerNormStyle.NO_LAYER_NORMALIZATION,
                 attention_type: AttentionType = \
                     AttentionType.DOT_PRODUCT
                 **kwargs  # type: ignore
                 ):
        super().__init__(**kwargs)  # type: ignore
        self.input_size = input_size
        self.output_size = ouptut_size
        self.hidden_size = hidden_size
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout_prob = classifier_dropout_prob
        self.layer_norm_style = use_layer_norm
        self.attention_type = attention_type

class Persformer(Module):
    
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
        return PersformerBlock(self.config)
                               
                        
        
    def forward(self,  # type: ignore
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
        # Apply the classifier layer
        output = self.classifier_layer(output)
        return output

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
            input_batch: The input batch. Of shape (batch_size, sequence_length, 2 + num_homology_types)
            attention_mask: The attention mask. Of shape (batch_size, sequence_length)
        
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
            input_batch: The input batch. Of shape (batch_size, sequence_length, 2 + num_homology_types)
            attention_mask: The attention mask. Of shape (batch_size, sequence_length)
        
        Returns:
            The logits of the model. Of shape (batch_size, sequence_length, 1)
        """
        output = self.attention_layer(input_batch, attention_mask) + input_batch
        output = self.feed_forward_layer(output) + output
        return output
        
    def _forward_pre_layer_norm(self,
                                input_batch: Tensor,
                                attention_mask: Optional[Tensor] = None
                                ) -> Tensor:
        """
        Forward pass of the model with pre-layer normalization.
        
        Args:
            input_batch: The input batch. Of shape (batch_size, sequence_length, 2 + num_homology_types)
            attention_mask: The attention mask. Of shape (batch_size, sequence_length)
        
        Returns:
            The logits of the model. Of shape (batch_size, sequence_length, 1)
        """
        assert self.layer_norms is not None
        normalized = self.layer_norms[0](input_batch)
        output = self.attention_layer(normalized, attention_mask) + input_batch
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
            input_batch: The input batch. Of shape (batch_size, sequence_length, 2 + num_homology_types)
            attention_mask: The attention mask. Of shape (batch_size, sequence_length)
        
        Returns:
            The logits of the model. Of shape (batch_size, sequence_length, 1)
        """
        assert self.layer_norms is not None
        output = self.attention_layer(input_batch, attention_mask) + input_batch
        output = self.layer_norms[0](output)
        output = self.feed_forward_layer(output) + output
        output = self.layer_norms[1](output)
        return output



def get_feed_forward_layer(config: PersformerConfig) -> Module:
    """
    Get the feed forward layer.
    """
    feed_forward_layer = Sequential()
    feed_forward_layer.add_module(
                                    "intermediate",
                                    Linear(config.hidden_size, config.intermediate_size)
                                )
    feed_forward_layer.add_module(
                                    "activation",
                                    get_activation_function(config.hidden_act)
                                )
    feed_forward_layer.add_module(
                                    "dropout",
                                    Dropout(config.hidden_dropout_prob)
                                )
    
    feed_forward_layer.add_module(
                                    "output",
                                    Linear(config.intermediate_size, config.hidden_size)
                                    )
    feed_forward_layer.add_module(
                                    "dropout",
                                    Dropout(config.hidden_dropout_prob)
                                )
    return feed_forward_layer


def get_attention_layer(config: PersformerConfig) -> Module:
    """
    Get the attention layer.
    """
    return AttentionFactory().build(config)


class AttentionBase(Module):
    def __init__(self, config: PersformerConfig):
        super().__init__()
        self.config = config
        
    def forward(self,  # type: ignore
                input: Tensor,
                attention_mask: Optional[Tensor] = None
                ) -> Tensor:
        raise NotImplementedError

class AttentionFactory():
    """
    Factory for creating attention modules.
    """
    attention_modules: Dict[AttentionType, Callable[[PersformerConfig], AttentionBase]] = {}
    
    def __init__(self):
            # Register the attention layers here:
            self.register_attention__builder(AttentionType.DOT_PRODUCT,
                                             lambda config: DotProductAttention(config))
            self.register_attention__builder(AttentionType.INDUCED_ATTENTION,
                                                lambda config: InducedAttention(config))
        
    
    def register_attention__builder(self,
                           attention_type: AttentionType,
                           attention_module_builder: Callable[[PersformerConfig], AttentionBase]) -> None:
        """
        Register an attention module.
        """
        self.attention_modules[attention_type] = attention_module_builder
        
    def build(self, config: PersformerConfig) -> AttentionBase:
            """
            Create an attention module.
            """
            return self.attention_modules[config.attention_type](config)
        
    

class DotProductAttention(AttentionBase):
    """
    Dot product attention. See https://arxiv.org/abs/1706.03762.
    """
    def __init__(self,
                 config: PersformerConfig):
        super().__init__(config)
        
        self.dot_product_attention = \
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
        attention_output, _ = self.dot_product_attention(input, input, input, attention_mask)
        return self.dropout(attention_output)

class InducedAttention(AttentionBase):
    def __init__(self, config: PersformerConfig) -> None:
        super().__init__(config)
        raise NotImplementedError
    
    def forward(self,  # type: ignore
                input: Tensor,
                attention_mask: Optional[Tensor] = None
                ):
        raise NotImplementedError
    