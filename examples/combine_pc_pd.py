
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
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup


from torch.utils.data import DataLoader, TensorDataset  # type: ignore

# Import the giotto-deep modules
from gdeep.data import OrbitsGenerator, DataLoaderKwargs
from gdeep.topology_layers import SetTransformer, PersFormer, SetTransformerOld
#from gdeep.topology_layers import AttentionPooling
from gdeep.topology_layers import ISAB, PMA, SAB
from gdeep.pipeline import Pipeline
from gdeep.search import Gridsearch
import json
#from gdeep.search import Gridsearch

from optuna.pruners import MedianPruner, NopPruner
from gdeep.search import VariationPruner

# %%

config_model = DotMap({
    'implementation': 'SetTransformerCombined', # SetTransformer, PersFormer,
    # PytorchTransformer, DeepSet, X-Transformer
    'dim_input': 2,
    'num_outputs': 1,  # for classification tasks this should be 1
    'num_classes': 5,  # number of classes
    'dim_hidden': 128,
    'num_heads': 8,
    'num_induced_points': 32,
    'layer_norm': False,  # use layer norm
    'simplified_layer_norm': False,  #Xu, J., et al. Understanding and improving layer normalization.
    'pre_layer_norm': False,
    'layer_norm_pooling': False,
    'num_layers_encoder': 2,
    'num_layers_decoder': 3,
    'attention_type': "self_attention",
    'activation': "gelu",
    'dropout_enc': 0.0,
    'dropout_dec': 0.2,
    'optimizer': Adam,
    'learning_rate': 1e-4,
    'num_epochs': 1000,
    'pooling_type': "attention",
    'weight_decay': 0.00,
    'n_accumulated_grads': 0,
    'bias_attention': "True",
    'warmup': 0.02,
})

#%%%
class SetTransformerCombined(nn.Module):
    """
    Set transformer architecture with old implementation.
    """
    def __init__(
        self,
        dim_input=4,  # dimension of input data for each element in the set
        num_outputs=1,
        dim_output=5,  # number of classes
        num_inds=32,  # number of induced points, see  Set Transformer paper
        dim_hidden=128,
        num_heads="4",
        layer_norm="False",  # use layer norm
        pre_layer_norm="False", # use pre-layer norm
        simplified_layer_norm="True",
        dropout_enc=0.0,
        dropout_dec=0.0,
        num_layer_enc=2,
        num_layer_dec=3,
        activation="gelu",
        bias_attention="True",
        attention_type="induced_attention",
        layer_norm_pooling="False",
    ):
        super().__init__()
        self._attention_type = attention_type
        bias_attention = eval(bias_attention)
        if activation == 'gelu':
            activation_layer = nn.GELU()
            activation_function = F.gelu
        elif activation == 'relu':
            activation_layer = nn.ReLU()
            activation_function = F.relu
        else:
            raise ValueError("Unknown activation '%s'" % activation)
        
        if attention_type=="induced_attention":
            self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, eval(num_heads), num_inds, ln=eval(layer_norm),
                        simplified_layer_norm = eval(simplified_layer_norm),
                        bias_attention=bias_attention, activation=activation),
                *[ISAB(dim_hidden, dim_hidden, eval(num_heads), num_inds, ln=eval(layer_norm),
                        simplified_layer_norm = eval(simplified_layer_norm),
                        bias_attention=bias_attention, activation=activation)
                    for _ in range(num_layer_enc-1)],
            )
        elif attention_type=="self_attention":
            self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, eval(num_heads), ln=eval(layer_norm),
                    simplified_layer_norm = eval(simplified_layer_norm),
                    bias_attention=bias_attention, activation=activation),
                *[SAB(dim_hidden, dim_hidden, eval(num_heads), ln=eval(layer_norm),
                        simplified_layer_norm = eval(simplified_layer_norm),
                        bias_attention=bias_attention, activation=activation)
                    for _ in range(num_layer_enc-1)],
            )
        elif attention_type=="pytorch_self_attention":
            emb = nn.Linear(dim_input, dim_hidden)
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim_hidden,
                                                    nhead=eval(num_heads),
                                                    dropout=dropout_enc,
                                                    activation=activation_function,
                                                    norm_first=eval(pre_layer_norm),
                                                    batch_first=True)
            self.enc = nn.Sequential(
                    emb,
                    nn.TransformerEncoder(encoder_layer,
                                                num_layers=num_layer_enc)
            )
        elif attention_type=="pytorch_self_attention_skip":
            self.emb = nn.Linear(dim_input, dim_hidden)
            self.encoder_layers = nn.ModuleList(
                [nn.TransformerEncoderLayer(d_model=dim_hidden,
                                            nhead=eval(num_heads),
                                            dropout=dropout_enc,
                                            activation=activation_function,
                                            norm_first=eval(pre_layer_norm),
                                            batch_first=True) for _ in range(num_layer_enc)]
            )
        else:
            raise ValueError("Unknown attention type: {}".format(attention_type))
        enc_layer_dim = [2**i if i <= num_layer_dec/2 else num_layer_dec - i for i in range(num_layer_dec)]
        self.pool = nn.Sequential(
            nn.Dropout(dropout_dec),
            PMA(dim_hidden, eval(num_heads), num_outputs, ln=eval(layer_norm_pooling),
                simplified_layer_norm = eval(simplified_layer_norm),
                bias_attention=bias_attention, activation=activation),
            nn.Dropout(dropout_dec),
        )

        self.ln = nn.LayerNorm(dim_hidden + 128)
        
        self.dec = nn.Sequential(
            *[nn.Sequential(nn.Linear(enc_layer_dim[i] * dim_hidden + 128, enc_layer_dim[i+1] * dim_hidden + 128),
                            activation_layer,
                            nn.Dropout(dropout_dec)) for i in range(num_layer_dec-1)],
            nn.Linear(enc_layer_dim[-1] * dim_hidden + 128, dim_output),
        )


    def forward(self, input, pd):
        if self._attention_type == "pytorch_self_attention_skip":
            x = self.emb(input)
            for l in self.encoder_layers:
                x = x + l(x)
        else:
            x = self.enc(input)
        return self.dec(self.ln(torch.cat((self.pool(x), 0.0 * pd), -1))).squeeze(dim=1)
# %%
pd = torch.load('data/orbit5k_pd_vec.pt')
x = torch.load('data/orbit5k_x.pt').float()
y = torch.load('data/orbit5k_y.pt')
dl_train = DataLoader(TensorDataset(x, y),
                                batch_size=32)


# %%
model = SetTransformerOld(dim_input=config_model.dim_input, dim_output=5,
                        num_inds=config_model.num_induced_points,
                        dim_hidden=config_model.dim_hidden,
                        num_heads=str(config_model.num_heads),
                        layer_norm=str(config_model.layer_norm),  # use layer norm
                        pre_layer_norm=str(config_model.pre_layer_norm),
                        simplified_layer_norm=str(config_model.simplified_layer_norm),
                        dropout_enc=config_model.dropout_enc,
                        dropout_dec=config_model.dropout_dec,
                        num_layer_enc=config_model.num_layers_encoder,
                        num_layer_dec=config_model.num_layers_decoder,
                        activation=config_model.activation,
                        bias_attention=config_model.bias_attention,
                        attention_type=config_model.attention_type,
                        layer_norm_pooling=str(config_model.layer_norm_pooling))


# %%
# initialise loss
loss_fn = nn.CrossEntropyLoss()

# Initialize the Tensorflow writer
#writer = SummaryWriter(comment=json.dumps(config_model.toDict())\
#                                + json.dumps(config_data.toDict()))
writer = SummaryWriter(comment=config_model.implementation)

optim = torch.optim.Adam(model.parameters(), 1e-3)

# initialise pipeline class
pipe = Pipeline(model, [dl_train, None], loss_fn, writer)
# %%


# train the model
pipe.train(config_model.optimizer,
           config_model.num_epochs,
           cross_validation=False,
           optimizers_param={"lr": config_model.learning_rate,
            "weight_decay": config_model.weight_decay},
           n_accumulated_grads=config_model.n_accumulated_grads,
           lr_scheduler=get_cosine_schedule_with_warmup,  #get_constant_schedule_with_warmup,  #get_cosine_with_hard_restarts_schedule_with_warmup,
           scheduler_params = {"num_warmup_steps": int(config_model.warmup * config_model.num_epochs),
                               "num_training_steps": config_model.num_epochs,},
                               #"num_cycles": 1},
           store_grad_layer_hist=False)

# pipe.train(config_model.optimizer,
#            config_model.num_epochs,
#            cross_validation=False,
#            optimizers_param={"lr": config_model.learning_rate,
#             "weight_decay": config_model.weight_decay},
#            store_grad_layer_hist=False)

# %%
from gdeep.models import ModelExtractor

loss_fn = nn.CrossEntropyLoss()
me = ModelExtractor(pipe.model, loss_fn)

list_set_representations = []

for x, y in dl_train:
    list_activations = me.get_activations(x)
    list_set_representations.append(list_activations[-4].detach().cpu())

torch.save(torch.cat(list_set_representations), 'data/orbit5k_pc_self_at.pt')

# %%