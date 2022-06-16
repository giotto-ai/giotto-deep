
from torch.nn import Module
from .persformer import Persformer
from .persformer_config import PersformerConfig
from gdeep.utility.enum_types import ActivationFunction, PoolerType, AttentionType, LayerNormStyle


class PersformerWrapper(Module):
    
    config: PersformerConfig
    model: Module
    
    def __init__(self,
                 **kwargs):
        super().__init__()
        self.config = PersformerConfig(
            **kwargs
        )
        self.model = Persformer(self.config)
    
    def forward(self, input, attention_mask):
        return self.model(input, attention_mask)