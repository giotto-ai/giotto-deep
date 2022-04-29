from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

from torch import layer_norm

from gdeep.topology_layers.residual import LayerNormStyle

@dataclass(init=False)
class PersformerEncoderConfig:
    """
    Configuration class to define a full Persformer model.
    
    
    
    """
    dim_model: int
    multi_head_config: Dict[str, Any]
    feedforward_config: Dict[str, Any]
    layer_norm_style: LayerNormStyle
    
    def __init__(
        self,
        dim_model: int,
        multi_head_config: Dict[str, Any],
        feedforward_config: Dict[str, Any],
        layer_norm_style: str = "post",
        ) -> None:
        
        try:
            if "dim_model" not in multi_head_config:
                multi_head_config["dim_model"] = dim_model
            if "dim_model" not in feedforward_config:
                feedforward_config["dim_model"] = dim_model
        except AttributeError:
            raise AttributeError("dim_model is required for all configs")
        
        
        self.dim_model = dim_model
        self.multi_head_config = multi_head_config
        self.feedforward_config = feedforward_config
        