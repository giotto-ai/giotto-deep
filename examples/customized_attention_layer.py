# Post-LN or Pre-LN: https://github.com/jwang0306/transformer-pytorch/blob/master/transformer.ipynb
#
#
# %%

import torch
from torch import Tensor, randn
from torch.nn import (Module, MultiheadAttention, Linear,
                      Sequential, LayerNorm, ReLU, Dropout,
                      Parameter)

pre_layer_norm: bool = True
dropout: float = 0.1

# %%




class AttentionLayer(Module):
    def __init__(self,
                 hidden_size: int=32,
                 filter_size: int=32,
                 n_heads: int=8,
                 layer_norm = True,
                 pre_layer_norm: bool=True,
                 dropout: float = 0.0,
                 activation=None,
                 attention_type='self_attention',
                 num_induced_points: int=32):
        super(AttentionLayer, self).__init__()
        
        if activation is None:
            self.activation = ReLU
        else:
            self.activation = activation
        self.dropout = dropout
        
        self.layer_norm = layer_norm
        
        # attention part
        self.fc_q = Linear(hidden_size, hidden_size)
        self.fc_k = Linear(hidden_size, hidden_size)
        self.fc_v = Linear(hidden_size, hidden_size)
        if attention_type == 'self_attention':
            self.attention_block = MultiheadAttention(hidden_size, n_heads)
        if attention_type == 'fast_attention':
            self.attention_block = FastAttention(hidden_size)
        else:
            raise NotImplementedError(r"{} attention is not implemented"
                                      .format(attention_type))
        if self.layer_norm:
            self.attention_ln = LayerNorm(hidden_size)
            #Alternate layer normal
            #self.attention_ln_q = LayerNorm(hidden_size)
            #self.attention_ln_kv = LayerNorm(hidden_size)
        
        # elementwise feed forward network
        self.eff = Sequential(
                              Linear(hidden_size, filter_size),
                              self.activation(inplace=True),
                              Dropout(self.dropout),
                              Linear(filter_size, hidden_size),
                              Dropout(dropout)
                              )
        
        if self.layer_norm:
            self.eff_ln = LayerNorm(hidden_size)
        
    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        q = self.fc_v(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        if self.pre_layer_norm:
            q_ln = self.attention_ln(q) if self.layer_norm else q
            k_ln = self.attention_ln(k) if self.layer_norm else k
            v_ln = self.attention_ln(v) if self.layer_norm else v
            #Alternate layer normal
            #q_ln = self.attention_ln_q(q) if self.layer_norm else q
            #kv_ln = self.attention_ln_kv(kv) if self.layer_norm else kv
            x = q + self.attention_block(q_ln, k_ln, v_ln)
            x_ln = self.eff_ln(x) if self.layer_norm else x
            x = x + self.eff(x_ln)
        else:
            x = q + self.attention_block(q, k, v)
            if self.layer_norm:
                x = self.attention_ln(x)
            x = x + self.eff(x)
            if self.layer_norm:
                x = self.eff_ln(x)
        
        return x

class InducedAttention(Module):
    def __init__(self,  num_induced_points: int=32):
        
        self.induced_points = Parameter(randn(num_induced_points, ))

    def forward(self, x: Tensor):
        pass


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
    
class AttentionPooling(Module):
    def __init__(self, hidden_dim, num_heads, num_seeds, ):
        super().__init__()
        self.Q = nn.Parameter(torch.Tensor(1, num_seeds, hidden_dim))
        nn.init.xavier_uniform_(self.S)
    
    
      

# %%
# %%
x = torch.rand(2, 10, 32)

mhsa = MultiheadAttention(32, 2)
model = LambdaLayer(lambda x: mhsa(x, x, x)[0]),
    

model(x)
# %%
mhsa(x, x, x)[0].shape
# %%
