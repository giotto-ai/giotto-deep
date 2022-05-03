from gdeep.topology_layers.feedforward import Feedforward, FeedforwardConfig

# Factory for creating models with a multi-head attention layer and a
# feedforward layer.

class FeedForwardFactory(object):
    def __init__(self, config: FeedforwardConfig) -> None:
        self.config = config
        self.feedforward = Feedforward(config)

    def __call__(self) -> Feedforward:
        return self.feedforward
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
     
    def __repr__(self) -> str:
        return str(self)

class MultiHeadAttentionFactory(object):
    def __init__(self, config: FeedforwardConfig) -> None:
        self.config = config
        self.multi_head_attention = MultiHeadAttention(config)

    def __call__(self) -> MultiHeadAttention:
        return self.multi_head_attention
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
     
    def __repr__(self) -> str:
        return str(self)
    
    
class MulitHeadAttentionBlockFactory(object):
    def __init__(self, config: FeedforwardConfig) -> None:
        self.config = config
        self.multi_head_attention_block = MultiHeadAttentionBlock(config)

    def __call__(self) -> MultiHeadAttentionBlock:
        return self.multi_head_attention_block
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
     
    def __repr__(self) -> str:
        return str(self)