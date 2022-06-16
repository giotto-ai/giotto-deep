from typing import Callable, Dict

from gdeep.utility.enum_types import AttentionType

from .attention import (AttentionBase, InducedAttention,
                        ScaledDotProductAttention)
from .persformer_config import PersformerConfig


class AttentionFactory():
    """
    Factory for creating attention modules.
    """
    attention_modules: Dict[AttentionType, Callable[[PersformerConfig], AttentionBase]] = {}
    
    def __init__(self):
            # Register the attention layers here:
            self.register_attention__builder(AttentionType.DOT_PRODUCT,
                                             lambda config: ScaledDotProductAttention(config))
            self.register_attention__builder(AttentionType.INDUCED_ATTENTION,
                                                lambda config: InducedAttention(config))
        
    
    def register_attention__builder(self,
                           attention_type: AttentionType,
                           attention_module_builder: Callable[[PersformerConfig], AttentionBase]
                           ) -> None:
        """
        Register an attention module.
        """
        self.attention_modules[attention_type] = attention_module_builder
        
    def build(self, config: PersformerConfig) -> AttentionBase:
            """
            Create an attention module.
            """
            return self.attention_modules[config.attention_type](config)
