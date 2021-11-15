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

# Import the giotto-deep modules
from gdeep.data import OrbitsGenerator, DataLoaderKwargs

# %%

dataloaders_dicts = DataLoaderKwargs(train_kwargs = {"batch_size": 32},
                                     val_kwargs = {"batch_size": 4},
                                     test_kwargs = {"batch_size": 3})

og = OrbitsGenerator(num_orbits_per_class=10,
                     homology_dimensions = (0, 1),
                     validation_percentage=0.0,
                     test_percentage=0.2)

dl_train, _, dl_test = og.get_dataloader_orbits(dataloaders_dicts)
# %%
class OrbitDataAugmenter(object):

    def __call__(self, x):
        
        return x
    
#%%
import numpy as np
p = 3.5
x = np.zeros((10, 2))
x[0] = np.random.rand(2)
for i in range(1, 10):  # type: ignore
    x_cur = x[i - 1, 0]
    y_cur = x[i - 1, 1]

    x[i, 0] = (x_cur + p * y_cur * (1. - y_cur)) % 1
    x_next = x[i, 0]
    x[i, 1] = (y_cur + p * x_next * (1. - x_next)) % 1

y = np.zeros((10, 2))
y[0] = np.array([1, 1]) - x[0]
for i in range(1, 10):  # type: ignore
    x_cur = y[i - 1, 0]
    y_cur = y[i - 1, 1]

    y[i, 0] = (x_cur + p * y_cur * (1. - y_cur)) % 1
    x_next = y[i, 0]
    y[i, 1] = (y_cur + p * x_next * (1. - x_next)) % 1
# %%
print(x)
print(np.array([1, 1]) - y)
# %%
