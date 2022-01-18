
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

# Import Tensorflow writer
from torch.utils.tensorboard import SummaryWriter  # type: ignore

# Import modules from XTransformers
#from x_transformers.x_transformers import AttentionLayers, Encoder, ContinuousTransformerWrapper
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

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

#Configs
config_data = DotMap({
    'batch_size_train': 32,
    'num_orbits_per_class': 20_000,
    'validation_percentage': 0.0,
    'test_percentage': 0.0,
    'num_jobs': 8,
    'dynamical_system': 'classical_convention',
    'homology_dimensions': (0, 1),
    'dtype': 'float32',
    'arbitrary_precision': False
})


config_model = DotMap({
    'implementation': 'Old_SetTransformer', # SetTransformer, PersFormer,
    # PytorchTransformer, DeepSet, X-Transformer
    'dim_input': 2 + len(config_data.homology_dimensions) if len(config_data.homology_dimensions) > 1 else 2,
    'num_outputs': 1,  # for classification tasks this should be 1
    'num_classes': 5,  # number of classes
    'dim_hidden': 64,
    'num_heads': 4,
    'num_induced_points': 32,
    'layer_norm': False,  # use layer norm
    'simplified_layer_norm': False,  #Xu, J., et al. Understanding and improving layer normalization.
    'pre_layer_norm': False,
    'layer_norm_pooling': False,
    'num_layers_encoder': 2,
    'num_layers_decoder': 3,
    'attention_type': "induced_attention",
    'activation': "gelu",
    'dropout_enc': 0.0,
    'dropout_dec': 0.0,
    'optimizer': Adam,
    'learning_rate': 1e-4,
    'num_epochs': 200,
    'pooling_type': "attention",
    'weight_decay': 0.00,
    'n_accumulated_grads': 0,
    'bias_attention': "True",
    'warmup': 0.02,
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

if len(config_data.homology_dimensions) == 0:
    dl_train, _, _ = og.get_dataloader_orbits(dataloaders_dicts)
else:
    dl_train, _, _ = og.get_dataloader_persistence_diagrams(dataloaders_dicts)



# Define the model    
if config_model.implementation == "Old_SetTransformer":
    # initialize SetTransformer model

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
else:
    raise Exception("Unknown Implementation")
# %%

if config_data.dtype == "float64":
    print("Use float64 model")
    model = model.double()
else:
    print("use float32 model")
    print(config_model)
    print(config_data)
    print(model)

# %%
# Do training and validation

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

pipe.train(config_model.optimizer,
           config_model.num_epochs,
           cross_validation=False,
           optimizers_param={"lr": config_model.learning_rate,
            "weight_decay": config_model.weight_decay},
           store_grad_layer_hist=False)

# %%
# keep training
#pipe.train(Adam, 100, False, keep_training=True, store_grad_layer_hist=False)

# %%
# %%
# Gridsearch

# initialise gridsearch
# pruner = NopPruner()
# search = Gridsearch(pipe, search_metric="accuracy", n_trials=50, best_not_last=True, pruner=pruner)

# dictionaries of hyperparameters
# optimizers_params = {"lr": [1e-3, 1e-0, None, True],
#                       "weight_decay": [0.0001, 0.2, None, True] }
# dataloaders_params = {"batch_size": [8, 16, 2]}
# models_hyperparams = {"n_layer_enc": [2, 4],
#                       "n_layer_dec": [1, 5],
#                       "num_heads": ["2", "4", "8"],
#                       "hidden_dim": ["16", "32", "64"],
#                       "dropout": [0.0, 0.5, 0.05],
#                       "layer_norm": ["True", "False"],
#                       "bias_attention": ["True", "False"]}#,
#                       #'pre_layer_norm': ["True", "False"]}
    
# scheduler_params = {"num_warmup_steps": int(0.1 * config_model.num_epochs),  #(int) – The number of steps for the warmup phase.
#                     "num_training_steps": config_model.num_epochs, #(int) – The total number of training steps.
#                     "num_cycles": [1, 3, 1]}

# # starting the gridsearch
# search.start((AdamW,), n_epochs=config_model.num_epochs, cross_validation=False,
#             optimizers_params=optimizers_params,
#             dataloaders_params=dataloaders_params,
#             models_hyperparams=models_hyperparams, lr_scheduler=get_cosine_with_hard_restarts_schedule_with_warmup,
#             scheduler_params=scheduler_params)


# %%
#print(search.best_val_acc_gs, search.best_val_loss_gs)
# %%
#df_res = search._results()
#df_res
#df_res.to_csv('set_transformer_grid_search.csv')
# %%
