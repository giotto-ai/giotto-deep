from enum import Enum, auto

class PoolerType(Enum):
    ATTENTION = auto()
    MAX = auto()
    MEAN = auto()
    SUM = auto()

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
    