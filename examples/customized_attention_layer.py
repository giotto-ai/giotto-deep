# Post-LN or Pre-LN: https://github.com/jwang0306/transformer-pytorch/blob/master/transformer.ipynb
#
#
# %%

import torch
from torch import Tensor, randn
from torch.nn import (Module, MultiheadAttention, Linear,
                      Sequential, LayerNorm, ReLU, Softmax, Dropout,
                      Parameter)
from torch.nn.init import xavier_uniform_

pre_layer_norm: bool = True
dropout: float = 0.1

# %%

class FastSelfAttention(Module):
    def __init__(self,
                hidden_size: int=32,
                n_heads: int=8):
        super(FastSelfAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (hidden_size, n_heads))
        self.attention_head_size = int(hidden_size % n_heads)
        self.num_attention_heads = n_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim= hidden_size
        
        self.query = Linear(self.input_dim, self.all_head_size)
        self.query_att = Linear(self.all_head_size, self.num_attention_heads)
        self.key = Linear(self.input_dim, self.all_head_size)
        self.key_att = Linear(self.all_head_size, self.num_attention_heads)
        self.transform = Linear(self.all_head_size, self.all_head_size)

        self.softmax = Softmax(dim=-1)
        
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, Linear) and module.bias is not None:
            module.bias.data.zero_()
                
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5


        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads*self.attention_head_size)
        pooled_query_repeat= pooled_query.repeat(1, seq_len,1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer=mixed_key_layer* pooled_query_repeat
        
        query_key_score=(self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
        

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        weighted_value =(pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer
      
        return weighted_value


class AttentionLayer(Module):
    def __init__(self,
                 hidden_size: int=32,
                 filter_size: int=32,
                 n_heads: int=8,
                 layer_norm = True,
                 pre_layer_norm: bool=True,
                 dropout: float = 0.0,
                 activation=None,
                 attention_type='self_attention'):
        super(AttentionLayer, self).__init__()
        
        if activation is None:
            self.activation = ReLU
        else:
            self.activation = activation
        self.dropout = dropout

        self.pre_layer_norm = pre_layer_norm
        self.layer_norm = layer_norm
        
        # attention part
        self.fc_q = Linear(hidden_size, hidden_size, bias=False)
        self.fc_k = Linear(hidden_size, hidden_size, bias=False)
        self.fc_v = Linear(hidden_size, hidden_size, bias=False)
        if attention_type == 'self_attention':
            self.attention_block = MultiheadAttention(hidden_size, n_heads)
        if attention_type == 'fast_attention':
            self.attention_block = FastSelfAttention(hidden_size)
        else:
            raise NotImplementedError(r"{} attention is not implemented"
                                      .format(attention_type))
        if self.layer_norm:
            self.attention_ln = LayerNorm(hidden_size)
            #Alternative layer normal
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
            # Layer norm of query, key and value
            # if they are the same, the three lines will compute the same
            q_ln = self.attention_ln(q) if self.layer_norm else q
            k_ln = self.attention_ln(k) if self.layer_norm else k
            v_ln = self.attention_ln(v) if self.layer_norm else v
            #Alternative layer normal
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
    def __init__(self,
                 dim_hidden: int=32,
                 num_induced_points: int=32,
                 *args, **kwargs):
        
        self.induced_points = Parameter(Tensor(1, num_induced_points, dim_hidden))
        xavier_uniform_(self.induced_points)
        self.attention_layer_1 = AttentionLayer(dim_hidden, *args, **kwargs)
        self.attention_layer_2 = AttentionLayer(dim_hidden, *args, **kwargs)

    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        h = self.attention_layer_1(
            self.induced_points.repeat(batch_size, 1, 1), x, x)
        return self.attention_layer_2(x, h, h)

class AttentionPooling(Module):
    def __init__(self, hidden_dim: int=32, q_length: int=1, *args, **kwargs):
        super().__init__()
        self.q = Parameter(Tensor(1, q_length, hidden_dim))
        xavier_uniform_(self.q)
        self.AttentionLayer(hidden_dim, *args, **kwargs)
        
    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        return self.attention_layer(self.q.repeat(batch_size, 1, 1), x, x)
    
    
      

# %%
# %%
x = torch.rand(2, 10, 32)

mhsa = MultiheadAttention(32, 2)
model = LambdaLayer(lambda x: mhsa(x, x, x)[0]),
    

model(x)
# %%
mhsa(x, x, x)[0].shape
# %%
