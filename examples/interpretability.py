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
from torch import nn

# Import Tensorflow writer
from torch.utils.tensorboard import SummaryWriter  # type: ignore


# Import the giotto-deep modules
from gdeep.data import OrbitsGenerator, DataLoaderKwargs
#from gdeep.topology_layers import AttentionPooling
from gdeep.topology_layers import ISAB, PMA
# %%
#Configs
config_data = DotMap({
    'batch_size_train': 128,
    'num_orbits_per_class': 1_000,
    'validation_percentage': 0.0,
    'test_percentage': 0.0,
    'num_jobs': 8,
    'dynamical_system': 'classical_convention',
    'homology_dimensions': (0, 1),
    'dtype': 'float32',
    'arbitrary_precision': False
})

dataloaders_dicts = DataLoaderKwargs(train_kwargs = {"batch_size":
                                                        config_data.batch_size_train,
                                                        "shuffle": False},
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

dl_train, _, _ = og.get_dataloader_persistence_diagrams(dataloaders_dicts)

# %% 
class SetTransformer(nn.Module):
    """ Vanilla SetTransformer from
    https://github.com/juho-lee/set_transformer/blob/master/main_pointcloud.py
    """
    def __init__(
        self,
        dim_input=3,  # dimension of input data for each element in the set
        num_outputs=1,
        dim_output=40,  # number of classes
        num_inds=32,  # number of induced points, see  Set Transformer paper
        dim_hidden=128,
        num_heads=4,
        ln=False,  # use layer norm
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, input):
        return self.dec(self.enc(input)).squeeze()

# %%%%%
model = torch.load("set_transformer_orbit5k_trained.pth")
model.eval()
# %%
x, y = next(iter(dl_train))

pred = torch.argmax(model(x.to('cuda')), -1)
# %%
from gdeep.models import ModelExtractor

loss_fn = nn.CrossEntropyLoss()
me = ModelExtractor(model, loss_fn)

list_x = []
list_y = []
list_set_representations = []

for x, y in dl_train:
    list_activations = me.get_activations(x)
    list_set_representations.append(list_activations[-4].detach().cpu())
    list_x.append(x)
    list_y.append(y)

torch.save(torch.cat(list_set_representations), 'data/orbit5k_pd_vec.pt')
torch.save(torch.tensor(og.get_orbits()), 'data/orbit5k_x.pt')
torch.save(torch.cat(list_y), 'data/orbit5k_y.pt')
# %%
import matplotlib.pyplot as plt

model.eval()
x, y = next(iter(dl_train))
x = x.to('cuda')

for i in [10, 12, 14, 15]:

    delta = torch.zeros_like(x[i].unsqueeze(0)).to('cuda')
    delta.requires_grad = True

    loss = model(x + delta)[i, y[i].item()]
    print(y[i].item())
    print(model(x + delta).shape)
    loss.backward()

    c = torch.sqrt((delta.grad.detach().cpu()**2).sum(axis=-1))


    #sc = plt.scatter(x[i, :, 0].cpu(), x[i, :, 1].cpu(), c=-torch.log(c_max - c + eps))
    sc = plt.scatter(x[i, :, 0].detach().cpu(), x[i, :, 1].detach().cpu(), c=c)

    plt.colorbar(sc)
    plt.savefig('plots/' + 'orbit5k' + str(y[i].item()) + '.pdf')
    plt.show()
    print('y:', y[i])
# %%
i = 15
max_pt_idx = torch.argmax(x[i, : , 1])


# %%

# Implementation of matplotlib function
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

import itertools

def flatten_list(list2d):
    return list(itertools.chain.from_iterable(list2d))

def plot_model(pd_x, class_idx):
    x_max = 0.2
    y_max = 0.2
    d = 0.05
    min_radius = 3.0 * d
    list2d_x = [[j * d for j in range(i + 1)] for i in range(int(x_max / d) + 1)]
    list2d_y = [[i * d for j in range(i + 1)] for i in range(int(x_max / d) + 1)]
    x = torch.tensor(flatten_list(list2d_x))
    y = torch.tensor(flatten_list(list2d_y))
    pd_x_expand = pd_x.unsqueeze(0).repeat(x.shape[0], 1, 1)
    pd_x_expand[:, max_pt_idx, :2] = torch.stack([x, y], 0).transpose(0, 1)
    pd_x_expand[:, max_pt_idx, :2]
    
    print("x", x.shape)
    print("y", y.shape)
    
    z = model(pd_x_expand.cuda()).detach().cpu()[:, class_idx]
    print(z.shape)
    
    triang = tri.Triangulation(x, y)
        
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    tcf = ax1.tricontourf(triang, z)
    fig1.colorbar(tcf)
    ax1.set_title('matplotlib.axes.Axes.tricontourf() Example')
    plt.show()
# %%
pd_x = x[i]
class_idx = y[i].item()
plot_model(pd_x, class_idx)


# %%
list2d_x = [[j for j in range(i + 1)] for i in range(int(2) + 1)]
list2d_y = [[i for j in range(i + 1)] for i in range(int(2) + 1)]
x = torch.tensor(flatten_list(list2d_x))
y = torch.tensor(flatten_list(list2d_y))
# %%
pd_x_expand = pd_x.unsqueeze(0).repeat(6, 1, 1)
pd_x_expand[:, max_pt_idx, :2] = torch.stack([x, y], 0).transpose(0, 1)
pd_x_expand[:, max_pt_idx, :2]
# %%

# %%
