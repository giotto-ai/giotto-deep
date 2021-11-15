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
#from gdeep.search import Gridsearch
# %%



# %%
# Define the data loader

homology_dimensions = (0, 1)

dataloaders_dicts = DataLoaderKwargs(train_kwargs = {"batch_size": 32},
                                     val_kwargs = {"batch_size": 4},
                                     test_kwargs = {"batch_size": 3})

og = OrbitsGenerator(num_orbits_per_class=1_000,
                     homology_dimensions = homology_dimensions,
                     validation_percentage=0.0,
                     test_percentage=0.0,
                     n_jobs=12
                     #dynamical_system = 'pp_convention'
                     )


dl_train, _, _ = og.get_dataloader_orbits(dataloaders_dicts)

# %%
# Define the model
model = SetTransformer(
        dim_input=2,
        num_outputs=1,  # for classification tasks this should be 1
        dim_output=5,  # number of classes
        dim_hidden=64,
        num_heads=4,
        num_inds=32,
        ln=False,  # use layer norm
        n_layers_encoder=2,
        n_layers_decoder=2,
        attention_type="self_attention",
        dropout=0.0
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
writer = SummaryWriter()

# initialise pipeline class
pipe = Pipeline(model, [dl_train, None], loss_fn, writer)
# %%


# train the model
pipe.train(Adam, 200, cross_validation=False, optimizers_param={"lr": 1e-4})

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


