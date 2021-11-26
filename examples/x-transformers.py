# %%
!pip install x-transformers

# %%
from IPython import get_ipython  # type: ignore

# %% 
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
# %%
import math
from dotmap import DotMap
import json

import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from inspect import isfunction
from collections import namedtuple

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# Import Tensorflow writer
from torch.utils.tensorboard import SummaryWriter  # type: ignore


from gdeep.topology_layers import AttentionPooling
from gdeep.data import OrbitsGenerator, DataLoaderKwargs
from gdeep.pipeline import Pipeline

from x_transformers.x_transformers import AttentionLayers, Encoder, ContinuousTransformerWrapper


# %%
#https://github.com/lucidrains/x-transformers/tree/55ca5d96c8b850b064177091f7a1dcfe784b24ce
# https://github.com/lucidrains/performer-pytorch
# %%
# def exists(val):
#     return val is not None

# def default(val, d):
#     if exists(val):
#         return val
#     return d() if isfunction(d) else d

# class TransformerWrapper(nn.Module):
#     def __init__(
#         self,
#         *,
#         input_dim=2,
#         attn_layers,
#         emb_dim = None,
#         max_mem_len = 0.,
#         shift_mem_down = 0,
#         emb_dropout = 0.,
#         num_memory_tokens = None,
#         tie_embedding = False
#     ):
#         super().__init__()
#         assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

#         dim = attn_layers.dim

#         self.emb = nn.Linear(input_dim, dim)

#         self.max_mem_len = max_mem_len
#         self.shift_mem_down = shift_mem_down

#         self.attn_layers = attn_layers
#         self.norm = nn.LayerNorm(dim)


#         # memory tokens (like [cls]) from Memory Transformers paper
#         num_memory_tokens = default(num_memory_tokens, 0)
#         self.num_memory_tokens = num_memory_tokens
#         if num_memory_tokens > 0:
#             self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))


#     def forward(
#         self,
#         x,
#         return_mems = False,
#         return_attn = False,
#         mems = None,
#         mask = None,
#         **kwargs
#     ):
#         b, n, _, device, num_mem = *x.shape, x.device, self.num_memory_tokens
#         x = self.emb(x)

#         if num_mem > 0:
#             mem = repeat(self.memory_tokens, 'n d -> b n d', b = b)
#             x = torch.cat((mem, x), dim = 1)

#         if self.shift_mem_down and exists(mems):
#             mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
#             mems = [*mems_r, *mems_l]

#         x, intermediates = self.attn_layers(x, mask = mask, mems = mems, return_hiddens = True, **kwargs)
#         x = self.norm(x)

#         mem, x = x[:, :num_mem], x[:, num_mem:]

#         out = x

#         if return_mems:
#             hiddens = intermediates.hiddens
#             new_mems = list(map(lambda pair: torch.cat(pair, dim = -2), zip(mems, hiddens))) if exists(mems) else hiddens
#             new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))
#             return out, new_mems

#         if return_attn:
#             attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
#             return out, attn_maps

#         return out

# %%

config_data = DotMap({
    'batch_size_train': 32,
    'num_orbits_per_class': 1_000,
    'validation_percentage': 0.0,
    'test_percentage': 0.0,
    'num_jobs': 12,
    'dynamical_system': 'classical_convention',
    'homology_dimensions': (0, 1)
})


config_model = DotMap({
    'implementation': 'X_transformers',
    'dim_input': 2,
    'num_outputs': 1,  # for classification tasks this should be 1
    'num_classes': 5,  # number of classes
    'dim_hidden': 64,
    'num_heads': 4,
    'num_induced_points': 32,
    'layer_norm': False,  # use layer norm
    'pre_layer_norm': True,
    'num_layers_encoder': 2,
    'num_layers_decoder': 2,
    'attention_type': "self_attention",
    'activation': nn.GELU,
    'dropout': 0.0,
    'batch_size_train': 32,
    'optimizer': torch.optim.Adam,
    'learning_rate': 1e-4,
    'num_epochs': 200,
})

model = \
nn.Sequential(
    ContinuousTransformerWrapper(
        dim_in = 2,
        use_pos_emb = True,
        max_seq_len = None,
        attn_layers = Encoder(
            dim = config_model.dim_hidden,
            depth = config_model.num_layers_encoder,
            heads = config_model.num_heads,
        ),
    ),
    AttentionPooling(hidden_dim = config_model.dim_hidden, q_length=1),
    nn.Sequential(*[nn.Linear(config_model.dim_hidden,
                        config_model.dim_hidden)
            for _ in range(config_model.num_layers_decoder)]),
    nn.Linear(config_model.dim_hidden, config_model.num_classes)
).cuda()

x = torch.rand(10, 256, 2).cuda()

out = model(x)
print('out shape:', out.shape)
# %%
dataloaders_dicts = DataLoaderKwargs(train_kwargs = {"batch_size":
                                                        config_data.batch_size_train,},
                                     val_kwargs = {"batch_size": 4},
                                     test_kwargs = {"batch_size": 3})

og = OrbitsGenerator(num_orbits_per_class=config_data.num_orbits_per_class,
                     homology_dimensions = config_data.homology_dimensions,
                     validation_percentage=config_data.validation_percentage,
                     test_percentage=config_data.test_percentage,
                     n_jobs=config_data.num_jobs,
                     dynamical_system = config_data.dynamical_system
                     )


dl_train, _, _ = og.get_dataloader_orbits(dataloaders_dicts)

# %%
# Do training and validation

# initialise loss
loss_fn = nn.CrossEntropyLoss()

# Initialize the Tensorflow writer
writer = SummaryWriter(comment=json.dumps(config_model.toDict())\
                                + json.dumps(config_data.toDict()))

# initialise pipeline class
pipe = Pipeline(model, [dl_train, None], loss_fn, writer)
# %%


# train the model
pipe.train(config_model.optimizer,
           config_model.num_epochs,
           cross_validation=False,
           optimizers_param={"lr": config_model.learning_rate})
# %%
