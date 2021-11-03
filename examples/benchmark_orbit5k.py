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
#from torch.utils.tensorboard import SummaryWriter  # type: ignore

# Import the giotto-deep modules
from gdeep.data import OrbitsGenerator, DataLoaderKwargs
from gdeep.topology_layers import SetTransformer
from gdeep.pipeline import Pipeline
from gdeep.search import Gridsearch
# %%
# Initialize the Tensorflow writer
writer = SummaryWriter()

# %%
# Define the data loader

homology_dimensions = (0, 1)

dataloaders_dicts = DataLoaderKwargs(train_kwargs = {"batch_size": 32},
                                     val_kwargs = {"batch_size": 4},
                                     test_kwargs = {"batch_size": 3})

og = OrbitsGenerator(num_orbits_per_class=10,
                     homology_dimensions = homology_dimensions,
                     validation_percentage=0.0,
                     test_percentage=0.2)

dl_train, _, dl_test = og.get_dataloader_orbits(dataloaders_dicts)

# %%
# Define the model
from gdeep.topology_layers import SetTransformer
model = SetTransformer(
            dim_input=len(homology_dimensions),
            dim_output=5,
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
pipe = Pipeline(model, [dl_train, dl_test], loss_fn, None)
# %%
pipe.train(Adam, 3, True, {"lr": 0.001})
# %%
for batch, (X, y) in enumerate(dl_train):
    print(X.shape)
# %%
from torch.utils.data.sampler import SubsetRandomSampler
tr_idx = list(range(len(dl_train)))
dl_tr = torch.utils.data.DataLoader(dl_train.dataset,
                                    shuffle=False,
                                    #pin_memory=True,
                                    batch_size=32,
                                    sampler=SubsetRandomSampler(tr_idx))
# %%
for batch, (X, y) in enumerate(dl_tr):
    print(X.shape)
# %%
tr_idx
# %%
