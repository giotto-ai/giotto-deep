# %%
from IPython import get_ipython

# %% 
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# %%
from gdeep.data import OrbitsGenerator
# %%
og = OrbitsGenerator(num_orbits_per_class=10,
                     validation_percentage=0.2,
                     test_percentage=0.2)
# %%
dl_train, dl_val, dl_test = og.get_dataloader_combined(batch_size=32)
# %%
for x, p, y in dl_train:
    print(x.shape)
    print(y.shape)
    break
# %%
og.get_persistence_diagrams().shape
# %%
