# %%
from IPython import get_ipython  # type: ignore

# %% 
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# %%
# Import the PyTorch modules
import torch  # type: ignore
from torch import nn  # type: ignore
from torch.optim import SGD, Adam, RMSprop  # type: ignore

# Import Tensorflow writer
from torch.utils.tensorboard import SummaryWriter  # type: ignore

# Import the giotto-deep modules
from gdeep.data import OrbitsGenerator, DataLoaderKwargs
from gdeep.topology_layers import SetTransformer, PersFormer, SmallDeepSet, PytorchTransformer
from gdeep.pipeline import Pipeline
import json
#from gdeep.search import Gridsearch
# %%

# Configs
config_data = {
    'num_orbits_per_class': 1_000,
    'validation_percentage': 0.0,
    'test_percentage': 0.0,
    'num_jobs': 12,
    'dynamical_system': 'classical_convention',
    'homology_dimensions': (0, 1)
}

config_model = {
    'implementation': 'SetTransformer',
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
    'optimizer': Adam,
    'learning_rate': 1e-4,
    'num_epochs': 200,
}

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
                     dynamical_system = config_data.dynamical_system
                     )


dl_train, _, _ = og.get_dataloader_orbits(dataloaders_dicts)

# %%



# Define the model
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


#model = PersFormer(
#            dim_input=2,
#            dim_output=5,
#            n_layers=5,
#            hidden_size=32,
#            n_heads=4,
#            dropout=0.1,
#            layer_norm=True,
#            pre_layer_norm=False,
#            activation=nn.GELU,
#            attention_layer_type="self_attention")

# model = PytorchTransformer(
#         dim_input=2,
#         dim_output=5,
#         hidden_size=64,
#         nhead=8,
#         activation='gelu',
#         norm_first=True,
#         num_layers=3,
#         dropout=0.0,
# ).double()

# %%
#small_model = SmallDeepSet(dim_input=2).double()

# %%
# Do training and validation

# initialise loss
loss_fn = nn.CrossEntropyLoss()

# Initialize the Tensorflow writer
writer = SummaryWriter(comment=json.dumps(config_model) + json.dumps(config_data))

# initialise pipeline class
pipe = Pipeline(model, [dl_train, None], loss_fn, writer)
# %%


# train the model
pipe.train(config_model.optimizer,
           config_model.num_epochs,
           cross_validation=False,
           optimizers_param={"lr": config_model.learning_rate})

# %%
# keep training
# pipe.train(Adam, 100, False, keep_training=True)

# %%
# For debugging
for batch, (X, y) in enumerate(dl_train):
    print(X.shape)
    print(model(X).shape)
    print(y)
    break

# %%
# %%


