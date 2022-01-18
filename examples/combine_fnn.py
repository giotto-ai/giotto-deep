
# %%
from IPython import get_ipython

# %%
from IPython import get_ipython  # type: ignore

# %% 
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# %%

from dotmap import DotMap
import json
import os

import numpy as np

# Import the PyTorch modules
import torch  # type: ignore
from torch import nn  # type: ignore
from torch.optim import SGD, Adam, RMSprop, AdamW  # type: ignore
import torch.nn as nn
import torch.nn.functional as F

# Import Tensorflow writer
from torch.utils.tensorboard import SummaryWriter  # type: ignore

# Import modules from XTransformers
#from x_transformers.x_transformers import AttentionLayers, Encoder, ContinuousTransformerWrapper
#from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup


from torch.utils.data import DataLoader, TensorDataset  # type: ignore

from gdeep.pipeline import Pipeline
import json
#from gdeep.search import Gridsearch

# %%

pd = torch.rand_like(torch.load('data/orbit5k_pd_vec.pt'))
pc = torch.load('data/orbit5k_pc_ind_at.pt').float()
y = torch.load('data/orbit5k_y.pt')
dl_train = DataLoader(TensorDataset(torch.cat((pd, pc), dim=-1).squeeze(1), y),
                                batch_size=32)

# %%

input_dim = 128 + 128
model =nn.Sequential(
    nn.LayerNorm(input_dim),
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 5)
)

# %%
# initialise loss
loss_fn = nn.CrossEntropyLoss()

# Initialize the Tensorflow writer
#writer = SummaryWriter(comment=json.dumps(config_model.toDict())\
#                                + json.dumps(config_data.toDict()))
writer = SummaryWriter()


# initialise pipeline class
pipe = Pipeline(model, [dl_train, None], loss_fn, writer)
# %%


# train the model
""" pipe.train(config_model.optimizer,
           config_model.num_epochs,
           cross_validation=False,
           optimizers_param={"lr": config_model.learning_rate,
            "weight_decay": config_model.weight_decay},
           n_accumulated_grads=config_model.n_accumulated_grads,
           lr_scheduler=get_cosine_schedule_with_warmup,  #get_constant_schedule_with_warmup,  #get_cosine_with_hard_restarts_schedule_with_warmup,
           scheduler_params = {"num_warmup_steps": int(config_model.warmup * config_model.num_epochs),
                               "num_training_steps": config_model.num_epochs,},
                               #"num_cycles": 1},
           store_grad_layer_hist=False) """

pipe.train(torch.optim.Adam,
           100,
           cross_validation=False,
           optimizers_param={"lr": 1e-3,
            "weight_decay": 0.0},
           store_grad_layer_hist=False)
# %%
