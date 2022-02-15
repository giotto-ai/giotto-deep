# %% [markdown]
#  ## Benchmarking PersFormer on the graph datasets.
#  We will compare the accuracy on the graph datasets of our SetTransformer
#  based on PersFormer with the perslayer introduced in the paper:
#  https://arxiv.org/abs/1904.09378
# %% [markdown]
#  ## Benchmarking MUTAG
#  We will compare the test accuracies of PersLay and PersFormer on the MUTAG
#  dataset. It consists of 188 graphs categorised into two classes.
#  We will train the PersFormer on the same input features as PersFormer to
#  get a fair comparison.
#  The features PersLay is trained on are the extended persistence diagrams of
#  the vertices of the graph filtered by the heat kernel signature (HKS)
#  at time t=10.
#  The maximum (wrt to the architecture and the hyperparameters) mean test
#  accuracy of PersLay is 89.8(±0.9) and the train accuracy with the same
#  model and the same hyperparameters is 92.3.
#  They performed 10-fold evaluation, i.e. splitting the dataset into
#  10 equally-sized folds and then record the test accuracy of the i-th
#  fold and training the model on the 9 other folds.
# %%
from IPython import get_ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Import libraries:
import os
import json
from dotmap import DotMap



import numpy as np

# Import the PyTorch modules
import torch  # type: ignore
from torch import nn  # type: ignore
from torch.optim import SGD, Adam, RMSprop, AdamW  # type: ignore
from torch.utils.data import TensorDataset, DataLoader

# Import Tensorflow writer
#from torch.utils.tensorboard import SummaryWriter  # type: ignore
from gdeep.search import GiottoSummaryWriter

from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

# Import the giotto-deep modules
from gdeep.topology_layers import Persformer
from gdeep.pipeline import Pipeline
from gdeep.search import Gridsearch
from gdeep.topology_layers import load_data_as_tensor, balance_binary_dataset,\
    print_class_balance

from optuna.pruners import MedianPruner, NopPruner

# %%

#Configs

model_data_file = 'model_data_specifications'

with open(os.path.join(model_data_file, 'Mutag_data.json')) as config_data_file:
    config_data = DotMap(json.load(config_data_file))


with open(os.path.join(model_data_file, 'Mutag_model.json')) as config_data_file:
    config_model = DotMap(json.load(config_data_file))
    

with open(os.path.join(model_data_file, 'Mutag_hyperparameter_space.json')) as config_data_file:
    hyperparameters_dicts = DotMap(json.load(config_data_file))
    dataloaders_params = hyperparameters_dicts.dataloaders_params
    models_hyperparams = hyperparameters_dicts.models_hyperparams
    optimizers_params = hyperparameters_dicts.optimizers_params
    schedulers_params = hyperparameters_dicts.schedulers_params
    
    

# %%
x_pds, y = load_data_as_tensor(config_data.dataset_name)  # type: ignore

# Balance labels in dataset

if config_data.balance_dataset:
    x_pds, y = balance_binary_dataset(x_pds, y, verbose=True)

print('class balance: {:.2f}'.format((y.sum() / y.shape[0]).item()))
# %%
# Set up dataset and dataloader

# create the datasets
graph_ds = TensorDataset(x_pds, y)

# Either use fixed train and validation split or use cross validation
if hyperparameters_dicts.cross_validation:
    graph_dl = DataLoader(
                        graph_ds,
                        num_workers=config_data.num_jobs,
                        batch_size=config_data.batch_size_train,
                        shuffle=True
                        )
else:
    # Split the dataset into training and validation
    total_size = x_pds.shape[0]
    train_size = int(total_size * config_data.train_percentage)
    graph_ds_train, graph_ds_val = torch.utils.data.random_split(
                                                        graph_ds,
                                                        [train_size,
                                                        total_size - train_size],
                                                        generator=torch.Generator().manual_seed(config_data.data_split_seed))


    # Define data loaders
    graph_dl_train = DataLoader(
        graph_ds_train,
        num_workers=config_data.num_jobs,
        batch_size=config_data.batch_size_train,
        shuffle=True
        )

    graph_dl_val = DataLoader(
        graph_ds_val,
        num_workers=config_data.num_jobs,
        batch_size=config_data.batch_size_val,
        shuffle=False
    )

    # Compute balance of train and validation datasets
        
    print_class_balance(graph_dl_train, 'train')
    print_class_balance(graph_dl_val, 'validation')

# %%
# Define and initialize the model
model = Persformer.from_config(config_model, config_data)


# %%
# Do training and validation

# initialize loss
loss_fn = nn.CrossEntropyLoss()

# Initialize the Tensorflow writer
writer = GiottoSummaryWriter(
            os.path.join("runs",
                        config_model.implementation +
                        "_" + config_data.dataset_name +
                        "_" + models_hyperparams.attention_type[0] +
                        "_" + "hyperparameter_search_giotto_4_best")
            )

# initialize pipeline object
if hyperparameters_dicts.cross_validation:
    pipe = Pipeline(model, [graph_dl, None], loss_fn, writer)
else:
    pipe = Pipeline(model, [graph_dl_train, graph_dl_val, None], loss_fn, writer)

# Use gradient clipping
if config_model.gradient_clipping == None:
    pipe.clip = 1.0  # use default clipping value 1.0
else:
    pipe.clip = config_model.gradient_clipping
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

# pipe.train(eval(config_model.optimizer),
#            config_model.num_epochs,
#            cross_validation=False,
#            optimizers_param={"lr": config_model.learning_rate,
#             "weight_decay": config_model.weight_decay},
#            store_grad_layer_hist=False)
# %%
# Hyperparameter search

pruner = NopPruner()
search = Gridsearch(pipe,
                    search_metric="accuracy",
                    n_trials=hyperparameters_dicts.n_trials,
                    best_not_last=True,
                    pruner=pruner)

#dictionaries of hyperparameters
# optimizers_params = {"lr": [1e-3, 1e-0, None, True],
#                       "weight_decay": [0.0001, 0.2, None, True] }
# dataloaders_params = {"batch_size": [8, 32, 2]}
# models_hyperparams = {"n_layer_enc": [2, 4, 1], #(int) - The number of layers in the encoder
#                       "n_layer_dec": [1, 5, 1], #(int) - The number of layers in the encoder
#                       "num_heads": ["2", "4", "8"], #(int) - The number of heads in the encoder
#                       "hidden_dim": ["16", "32", "64", "96", "128"], #(int) - The number of hidden dimensions in the encoder
#                       "dropout_enc": [0.0, 0.5, 0.05],
#                       "dropout_dec": [0.0, 0.5, 0.05], 
#                       "layer_norm": ["True", "False"],
#                       "pre_layer_norm": ["True", "False"],
#                       "bias_attention": ["True", "False"],
#                       "input_dim": [config_model["input_dim"]],
#                       "pooling_type": ["pytorch_self_attention_skip"],
#                       "layer_norm_pooling": ["True", "False"],
#                       "activation": ["gelu",]
#                       }

# schedulers_params = {"num_warmup_steps": [int(0.02 * config_model.num_epochs)],  #(int) – The number of steps for the warmup phase.
#                     "num_training_steps": [config_model.num_epochs], #(int) – The total number of training steps.
#                     "num_cycles": [1]} #(int) – The number of restart cycles
#%%
# starting the gridsearch
search.start((eval(config_model.optimizer),),
            n_epochs=schedulers_params.num_training_steps[0],
            cross_validation=hyperparameters_dicts.cross_validation,
            k_folds=hyperparameters_dicts.k_folds,
            optimizers_params=optimizers_params,
            dataloaders_params=dataloaders_params,
            models_hyperparams=models_hyperparams, lr_scheduler=get_cosine_with_hard_restarts_schedule_with_warmup,
            schedulers_params=schedulers_params)

# %%
# from gdeep.visualisation import plotly2tensor
# from plotly.io import write_image
# import plotly.express as px
# df = px.data.iris()

# fig = px.scatter(
#     df, x="sepal_width", y="sepal_length", color="species"
# )
# write_image(fig, "deleteme.jpeg", format="jpeg", engine="orca")
# fig.show()
# plotly2tensor(fig)
# %%
import matplotlib.pyplot as plt

search.model.eval()
x, y = next(iter(graph_dl))

for i in range(1):

    delta = torch.zeros_like(x[i].unsqueeze(0))
    delta.requires_grad = True

    loss = search.model(x[i].unsqueeze(0) + delta)[0, y[i].item()]
    loss.backward()

    c = torch.sqrt((delta.grad[:, :, :2].detach().cpu()**2).sum(axis=-1))


    #sc = plt.scatter(x[i, :, 0].cpu(), x[i, :, 1].cpu(), c=-torch.log(c_max - c + eps))
    sc = plt.scatter(x[i, :, 0].detach(), x[i, :, 1].detach(), c=torch.nn.functional.normalize(c))

    plt.colorbar(sc)
    plt.plot([0, x[i, :, 0].max()], [0, x[i, :, 1].max()])
    plt.savefig('plots/' + 'orbit5k' + str(y[i].item()) + '.pdf')
    plt.show()
    print('y:', y[i])
# %%
