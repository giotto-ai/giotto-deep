# %%
from IPython import get_ipython

# %% 
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# %%
import torch
from gdeep.data import OrbitsGenerator
# %%
from sklearn.model_selection import train_test_split

ds_size = 10
val_size = 0.2
test_size = 0.2

idcs = torch.arange(10)

rest_idcs, test_idcs = train_test_split(idcs, test_size=test_size)
train_idcs, val_idcs = train_test_split(rest_idcs,
                                        test_size =
                                        val_size / (1.0 - test_size))
# %%
