# Post-LN or Pre-LN: https://github.com/jwang0306/transformer-pytorch/blob/master/transformer.ipynb
#
#
# %%

import torch
from torch import Tensor, randn
from torch.nn import (Module, MultiheadAttention, Linear,
                      Sequential, LayerNorm, ReLU, Softmax, Dropout,
                      Parameter, ModuleList)
from torch.nn.init import xavier_uniform_

pre_layer_norm: bool = True
dropout: float = 0.1

# %%

    
    


# %%
# %%
x = torch.rand(2, 10, 32)
# %%
