

from .persformer import Persformer 
class PersformerWrapper(Module):
    def __init__(self,
                 input_size: int = 2 + 4,
                 ouptut_size: int = 2,
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
                     AttentionType.DOT_PRODUCT,
                 pooler_type: PoolerType = PoolerType.ATTENTION)
    self.config = PersformerConfig(
        input_size=input_size,
        ouptut_size=ouptut_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_attention_layers=num_attention_layers,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        layer_norm_eps=layer_norm_eps,
        classifier_dropout_prob=classifier_dropout_prob,
        use_layer_norm=use_layer_norm,
        attention_type=attention_type,
        pooler_type=pooler_type,
    )
    self.model = Persformer(self.config)
    
    def forward(self, input, attention_mask):
        return self.model(input, attention_mask)