# from https://github.com/juho-lee/set_transformer/blob/master/modules.py

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch import einsum  # type: ignore
import math
from einops import rearrange # type: ignore


class _MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, bias_attention=True, activation='gelu',
                 simplified_layer_norm=True):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V, bias=bias_attention)
        self.fc_k = nn.Linear(dim_K, dim_V, bias=bias_attention)
        self.fc_v = nn.Linear(dim_K, dim_V, bias=bias_attention)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V, elementwise_affine = not simplified_layer_norm)
            self.ln1 = nn.LayerNorm(dim_V, elementwise_affine = not simplified_layer_norm)
        self.fc_o = nn.Linear(dim_V, dim_V, bias=True)
        if activation == 'gelu':
            self.activation_function = nn.GELU()
        elif activation == 'relu':
            self.activation_function = nn.ReLU()
        else:
            raise ValueError("Unknown activation '%s'" % activation)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        if hasattr(self, 'activation_function'):
            O = O + self.activation_function(self.fc_o(O))
        else:
            O = O + nn.ReLU()(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class _SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, bias_attention=True,
                 activation='gelu', simplified_layer_norm=True):
        super().__init__()
        self.mab = _MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, bias_attention=bias_attention,
                       activation=activation, simplified_layer_norm=simplified_layer_norm)

    def forward(self, X):
        return self.mab(X, X)


class _ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, bias_attention=True,
                 activation='gelu', simplified_layer_norm=True):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = _MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, bias_attention=bias_attention,
                        activation=activation, simplified_layer_norm=simplified_layer_norm)
        self.mab1 = _MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, bias_attention=bias_attention,
                        activation=activation, simplified_layer_norm=simplified_layer_norm)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class _PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, bias_attention=True,
                 activation='gelu', simplified_layer_norm=True):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = _MAB(dim, dim, dim, num_heads, ln=ln, bias_attention=bias_attention,
                       activation=activation, simplified_layer_norm=simplified_layer_norm)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class _FastAttention(nn.Module):
    """https://github.com/lucidrains/fast-transformer-pytorch/blob/main/fast_transformer_pytorch/fast_transformer_pytorch.py"""
    def __init__(
        self,
        input_dim,
        output_dim,
        *,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias = False)



        self.to_q_attn_logits = nn.Linear(dim_head, 1, bias = False)  # for projecting queries to query attention logits
        self.to_k_attn_logits = nn.Linear(dim_head, 1, bias = False)  # for projecting keys to key attention logits

        # final transformation of values to "r" as in the paper

        self.to_r = nn.Linear(dim_head, dim_head)

        self.to_out = nn.Linear(inner_dim, output_dim)

    def forward(self, x):
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)


        q_aggr, k_aggr, v_aggr = q, k, v

        # calculate query attention logits

        q_attn_logits = rearrange(self.to_q_attn_logits(q), 'b h n () -> b h n') * self.scale
        q_attn = q_attn_logits.softmax(dim = -1)

        # calculate global query token

        global_q = einsum('b h n, b h n d -> b h d', q_attn, q_aggr)
        global_q = rearrange(global_q, 'b h d -> b h () d')

        # bias keys with global query token

        k = k * global_q


        # now calculate key attention logits

        k_attn_logits = rearrange(self.to_k_attn_logits(k), 'b h n () -> b h n') * self.scale
        k_attn = k_attn_logits.softmax(dim = -1)

        # calculate global key token

        global_k = einsum('b h n, b h n d -> b h d', k_attn, k_aggr)
        global_k = rearrange(global_k, 'b h d -> b h () d')

        # bias the values

        u = v_aggr * global_k


        # transformation step

        r = self.to_r(u)

        # paper then says to add the queries as a residual

        r = r + q

        # combine heads

        r = rearrange(r, 'b h n d -> b n (h d)')
        return self.to_out(r)
