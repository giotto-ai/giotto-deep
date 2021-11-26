# %%
# for gridsearch

#!pip install pyyaml==5.4.1

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
from torch.optim import SGD, Adam, RMSprop  # type: ignore

# Import Tensorflow writer
from torch.utils.tensorboard import SummaryWriter  # type: ignore

# Import modules from XTransformers
from x_transformers.x_transformers import AttentionLayers, Encoder, ContinuousTransformerWrapper


# Import the giotto-deep modules
from gdeep.data import OrbitsGenerator, DataLoaderKwargs
from gdeep.topology_layers import SetTransformer, PersFormer, DeepSet, PytorchTransformer
from gdeep.topology_layers import AttentionPooling
from gdeep.pipeline import Pipeline
from gdeep.search import Gridsearch
import json
#from gdeep.search import Gridsearch

from optuna.pruners import MedianPruner, NopPruner

# %%

#Configs
config_data = DotMap({
    'batch_size_train': 32,
    'num_orbits_per_class': 1_000,
    'validation_percentage': 0.0,
    'test_percentage': 0.0,
    'num_jobs': 2,
    'dynamical_system': 'classical_convention',
    'homology_dimensions': (0, 1),
    'dtype': 'float32',
    'arbitrary_precision': True
})


config_model = DotMap({
    'implementation': 'SetTransformer', # SetTransformer, PersFormer,
    # PytorchTransformer, DeepSet, X-Transformer
    'dim_input': 2,
    'num_outputs': 1,  # for classification tasks this should be 1
    'num_classes': 5,  # number of classes
    'dim_hidden': 64,
    'num_heads': 4,
    'num_induced_points': 32,
    'layer_norm': False,  # use layer norm
    'pre_layer_norm': False,
    'num_layers_encoder': 3,
    'num_layers_decoder': 3,
    'attention_type': "self_attention",
    'activation': nn.GELU,
    'dropout': 0.0,
    'batch_size_train': 32,
    'optimizer': torch.optim.Adam,
    'learning_rate': 5e-4,
    'num_epochs': 10,
    'pooling_type': "max"
})



# %%



# Define the data loader


dataloaders_dicts = DataLoaderKwargs(train_kwargs = {"batch_size":
                                                        config_data.batch_size_train,},
                                     val_kwargs = {"batch_size": 4},
                                     test_kwargs = {"batch_size": 3})

og = OrbitsGenerator(num_orbits_per_class=config_data.num_orbits_per_class,
                     homology_dimensions = config_data.homology_dimensions,
                     validation_percentage=config_data.validation_percentage,
                     test_percentage=config_data.test_percentage,
                     n_jobs=config_data.num_jobs,
                     dynamical_system = config_data.dynamical_system,
                     dtype=config_data.dtype,
                     arbitrary_precision=config_data.arbitrary_precision,
                     )

if config_data.arbitrary_precision:
    orbits = np.load(os.path.join('data', 'orbit5k_arbitrary_precision.npy'))
    og.orbits_from_array(orbits)

dl_train, _, _ = og.get_dataloader_orbits(dataloaders_dicts)

# %%
import numpy as np
orbits = og.get_orbits()

np.save('orbit5k_arbitrary_precision.npy', orbits)
#%%



# Define the model
if config_model.implementation == 'SetTransformer':
    model = SetTransformer(
            dim_input=config_model.dim_input,
            num_outputs=1,  # for classification tasks this should be 1
            dim_output=config_model.num_classes,  # number of classes
            dim_hidden=config_model.dim_hidden,
            num_heads=config_model.num_heads,
            num_inds=config_model.num_induced_points,
            ln=config_model.layer_norm,  # use layer norm
            n_layers_encoder=config_model.num_layers_encoder,
            n_layers_decoder=config_model.num_layers_decoder,
            attention_type=config_model.attention_type,
            dropout=config_model.dropout
    )

elif config_model.implementation == 'PersFormer':
    model = PersFormer(
            dim_input=2,
            dim_output=5,
            n_layers=5,
            hidden_size=32,
            n_heads=4,
            dropout=0.1,
            layer_norm=True,
            pre_layer_norm=False,
            activation=nn.GELU,
            attention_layer_type="self_attention")

elif config_model.implementation == 'PytorchTransformer':
    model = PytorchTransformer(
            dim_input=2,
            dim_output=5,
            hidden_size=64,
            nhead=8,
            activation='gelu',
            norm_first=True,
            num_layers=3,
            dropout=0.0,
    )
elif config_model.implementation == 'DeepSet':
    model = DeepSet(dim_input=2,
                    dim_output=config_model.num_classes,
                    dim_hidden=config_model.dim_hidden,
                    n_layers_encoder=config_model.num_layers_encoder,
                    n_layers_decoder=config_model.num_layers_decoder,
                    pool=config_model.pooling_type).double()

elif config_model.implementation == "X-Transformer":
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
        nn.Sequential(*[nn.Sequential(nn.Linear(config_model.dim_hidden,
                            config_model.dim_hidden),
                            nn.ReLU())
                for _ in range(config_model.num_layers_decoder)]),
        nn.Linear(config_model.dim_hidden, config_model.num_classes)
    )

else:
    raise Exception("Unknown Implementation")
# %%
class SmallDeepSet(nn.Module):
    def __init__(self,
        pool="max",
        dim_input=2,
        dim_output=5,):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=dim_input, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=dim_output),
        )
        self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        x = self.dec(x)
        return x

if config_data.dtype == "float64":
    print("Use float64 model")
    model = model.double()
else:
    print("use float32 model")

# %%
# Do training and validation

# initialise loss
loss_fn = nn.CrossEntropyLoss()

# Initialize the Tensorflow writer
#writer = SummaryWriter(comment=json.dumps(config_model.toDict())\
#                                + json.dumps(config_data.toDict()))
writer = SummaryWriter(comment=config_model.implementation)

# initialise pipeline class
pipe = Pipeline(model, [dl_train, None], loss_fn, writer)
# %%


# train the model
# pipe.train(config_model.optimizer,
#            config_model.num_epochs,
#            cross_validation=False,
#            optimizers_param={"lr": config_model.learning_rate})

# %%
# keep training
#pipe.train(Adam, 300, False, keep_training=True)

# %%
# %%
# Gridsearch

# initialise gridsearch
search = Gridsearch(pipe, search_metric="accuracy", n_trials=10, best_not_last=True, pruner=NopPruner)

# dictionaries of hyperparameters
optimizers_params = {"lr": [1e-6, 1e-3]}
dataloaders_params = {"batch_size": [32, 64, 16]}
models_hyperparams = {"n_layer_enc": [2, 5],
                      "n_layer_dec": [2, 4],
                      "num_heads": ["2", "4", "8"],
                      "hidden_dim": ["16", "32", "64"],
                      "dropout": [0.0, 0.2],
                      "layer_norm": ["True", "False"],
                      'pre_layer_norm': ["True", "False"]}

# starting the gridsearch
search.start((Adam,), n_epochs=300, cross_validation=False,
             optimizers_params=optimizers_params,
             dataloaders_params=dataloaders_params,
             models_hyperparams=models_hyperparams, lr_scheduler=None,
             scheduler_params=None)


# %%
print(search.best_val_acc_gs, search.best_val_loss_gs)
# %%
df_res = search._results()
df_res
# %%
