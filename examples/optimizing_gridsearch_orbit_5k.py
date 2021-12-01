# for gridsearch


from dotmap import DotMap
import numpy as np

# Import the PyTorch modules
import torch  # type: ignore
from torch import nn  # type: ignore
from torch.optim import SGD, Adam, RMSprop  # type: ignore

# Import Tensorflow writer
from torch.utils.tensorboard import SummaryWriter  # type: ignore


# Import the giotto-deep modules
from gdeep.data import OrbitsGenerator, DataLoaderKwargs
from gdeep.topology_layers import ISAB, PMA, SAB
from gdeep.pipeline import Pipeline
from gdeep.search import Gridsearch

from optuna.pruners import MedianPruner, NopPruner


# Data and model configuration
config_data = DotMap({
    'batch_size_train': 32,
    'num_orbits_per_class': 2_000,
    'validation_percentage': 0.0,
    'test_percentage': 0.0,
    'num_jobs': 2,
    'dynamical_system': 'classical_convention',
    'homology_dimensions': (0, 1),
    'dtype': 'float32',
    'arbitrary_precision': False
})


config_model = DotMap({
    'implementation': 'Old_SetTransformer',
    'dim_input': 4,
    'num_outputs': 1,  # for classification tasks this should be 1
    'num_classes': 5,  # number of classes
    'dim_hidden': 64,
    'num_heads': 4,
    'num_induced_points': 64,
    'layer_norm': False,  # use layer norm
    'pre_layer_norm': False,
    'num_layers_encoder': 2,
    'num_layers_decoder': 1,
    'attention_type': "induced_attention",
    'activation': "nn.ReLU()",
    'dropout_enc': 0.0,
    'dropout_dec': 0.0,
    'optimizer': torch.optim.Adam,
    'learning_rate': 1e-4,
    'num_epochs': 500,
    'pooling_type': "attention",
    'weight_decay': 0.0,
    'n_accumulated_grads': 2,
    'bias_attention': "True"
})




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

# Define the data loader either consisting of the point clouds or the persistence diagrams
if config_data.dim_input == 2:
    dl_train, _, _ = og.get_dataloader_orbits(dataloaders_dicts)
else:
    dl_train, _, _ = og.get_dataloader_persistence_diagrams(dataloaders_dicts)



# Define the model
if config_model.implementation == "Old_SetTransformer":
    # initialize SetTransformer model
    class SetTransformerOld(nn.Module):
        def __init__(
            self,
            dim_input=4,  # dimension of input data for each element in the set
            num_outputs=1,
            dim_output=5,  # number of classes
            num_inds=32,  # number of induced points, see  Set Transformer paper
            dim_hidden=128,
            num_heads="4",
            layer_norm="False",  # use layer norm
            dropout=0.0,
            num_layer_enc=2,
            num_layer_dec=2,
            activation="nn.Relu()",
            bias_attention="True",
            attention_type="induced_attention"
        ):
            super().__init__()
            bias_attention = eval(bias_attention)
            if attention_type=="induced_attention":
                self.enc = nn.Sequential(
                    ISAB(dim_input, dim_hidden, eval(num_heads), num_inds, ln=eval(layer_norm), bias_attention=bias_attention),
                    *[ISAB(dim_hidden, dim_hidden, eval(num_heads), num_inds, ln=eval(layer_norm), bias_attention=bias_attention)
                      for _ in range(num_layer_enc-1)],
                )
            else:
                self.enc = nn.Sequential(
                    SAB(dim_input, dim_hidden, eval(num_heads), ln=eval(layer_norm), bias_attention=bias_attention),
                    *[SAB(dim_hidden, dim_hidden, eval(num_heads), ln=eval(layer_norm), bias_attention=bias_attention)
                      for _ in range(num_layer_enc-1)],
                )
            self.dec = nn.Sequential(
                nn.Dropout(dropout),
                PMA(dim_hidden, eval(num_heads), num_outputs, ln=eval(layer_norm), bias_attention=bias_attention),
                nn.Dropout(dropout),
                *[nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                nn.ReLU(),
                                nn.Dropout(dropout)) for _ in range(num_layer_dec-1)],
                nn.Linear(dim_hidden, dim_output),
            )

        def forward(self, input):
            return self.dec(self.enc(input)).squeeze(dim=1)



    model = SetTransformerOld(dim_input=4, dim_output=5,
                           num_inds=config_model.num_induced_points,
                           dim_hidden=config_model.dim_hidden,
                           num_heads=str(config_model.num_heads),
                           layer_norm=str(config_model.layer_norm),  # use layer norm
                           dropout=config_model.dropout_dec,
                           num_layer_enc=config_model.num_layers_encoder,
                           num_layer_dec=config_model.num_layers_decoder,
                           activation=config_model.activation,
                           bias_attention=config_model.bias_attention,
                           attention_type=config_model.attention_type)
else:
    raise Exception("Unknown Implementation")

# Use the model with the floating point precision given in configuration.
if config_data.dtype == "float64":
    print("Use float64 model")
    model = model.double()
else:
    print("Use float32 model")
    print(config_model)
    print(config_data)
    print(model)

# Do training and validation

# initialise loss
loss_fn = nn.CrossEntropyLoss()

# Initialize the Tensorflow writer
writer = SummaryWriter(comment=config_model.implementation)

# initialise pipeline class
pipe = Pipeline(model, [dl_train, None], loss_fn, writer)

# Gridsearch

# initialise gridsearch
pruner = NopPruner()
search = Gridsearch(pipe, search_metric="accuracy", n_trials=50, best_not_last=True, pruner=pruner)

# dictionaries of hyperparameters
optimizers_params = {"lr": [1e-7, 1e-3, None, True]}#,
                      #"weight_decay": [0.0, 0.2] }
dataloaders_params = {"batch_size": [16, 32, 4]}
models_hyperparams = {"n_layer_enc": [2, 3],
                      "n_layer_dec": [1, 5],
                      "num_heads": ["2", "4", "8"],
                      "hidden_dim": ["16", "32", "64"],
                      "dropout": [0.0, 0.5, 0.25],
                      "layer_norm": ["True", "False"],
                      "bias_attention": ["True", "False"]}

# starting the gridsearch
search.start((Adam,), n_epochs=1_000, cross_validation=False,
            optimizers_params=optimizers_params,
            dataloaders_params=dataloaders_params,
            models_hyperparams=models_hyperparams, lr_scheduler=None,
            scheduler_params=None, n_accumulated_grads=config_model.n_accumulated_grads)


# Save gridsearch results to csv file
df_res = search._results()
df_res.to_csv('set_transformer_grid_search.csv')