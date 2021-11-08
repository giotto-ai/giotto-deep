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
from gdeep.topology_layers import SetTransformer, PersFormer
from gdeep.pipeline import Pipeline
#from gdeep.search import Gridsearch
# %%
# Initialize the Tensorflow writer
writer = SummaryWriter()

#%%
%load_ext tensorboard
# %%
%tensorboard --logdir=examples/runs --port 6060

# %%
# Define the data loader

homology_dimensions = (0, 1)

dataloaders_dicts = DataLoaderKwargs(train_kwargs = {"batch_size": 32},
                                     val_kwargs = {"batch_size": 4},
                                     test_kwargs = {"batch_size": 3})

og = OrbitsGenerator(num_orbits_per_class=5000,
                     homology_dimensions = homology_dimensions,
                     validation_percentage=0.0,
                     test_percentage=0.2)

dl_train, _, dl_test = og.get_dataloader_orbits(dataloaders_dicts)

# %%
# Define the model
model = PersFormer(
            dim_input=len(homology_dimensions),
            dim_output=5,
            dropout=0.05,
            n_layers=3,
            ln=True,
            attention_type="induced_attention").double()
# %%
for x, y in dl_train:
    print(x.shape)
    print(model(x).shape)


# %%
# Do training and validation

# initialise loss
loss_fn = nn.CrossEntropyLoss()

# initialise pipeline class
pipe = Pipeline(model, [dl_train, dl_test], loss_fn, writer)
# %%
pipe.train(Adam, 100, cross_validation=False, optimizers_param={"lr": 0.001})
# %%

for batch, (X, y) in enumerate(dl_train):
    print(X.shape)
