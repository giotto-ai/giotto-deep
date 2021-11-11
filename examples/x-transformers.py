# %%
!pip install x-transformers
# %%
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from inspect import isfunction
from collections import namedtuple

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from x_transformers.x_transformers import *

# %%
#https://github.com/lucidrains/x-transformers/tree/55ca5d96c8b850b064177091f7a1dcfe784b24ce
# https://github.com/lucidrains/performer-pytorch
# %%
class TransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        input_dim=2,
        attn_layers,
        emb_dim = None,
        max_mem_len = 0.,
        shift_mem_down = 0,
        emb_dropout = 0.,
        num_memory_tokens = None,
        tie_embedding = False
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim

        self.emb = nn.Linear(input_dim, dim)

        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)


        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))


    def forward(
        self,
        x,
        return_mems = False,
        return_attn = False,
        mems = None,
        mask = None,
        **kwargs
    ):
        b, n, _, device, num_mem = *x.shape, x.device, self.num_memory_tokens
        x = self.emb(x)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, 'n d -> b n d', b = b)
            x = torch.cat((mem, x), dim = 1)

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, return_hiddens = True, **kwargs)
        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = x

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim = -2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))
            return out, new_mems

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out

# %%
model = TransformerWrapper(
    attn_layers = Encoder(
        dim = 512,
        depth = 12,
        heads = 8
    )
).cuda()

x = torch.rand(10, 256, 2).cuda()

out = model(x)
print('out shape:', out.shape)
# %%
