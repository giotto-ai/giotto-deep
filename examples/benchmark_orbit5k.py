# %%
from IPython import get_ipython

# %% 
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# %%
from gdeep.data import OrbitsGenerator, DataLoaderKwargs
from gdeep.topology_layers import SetTransformer
# %%
homology_dimensions = (0, 1)

dataloaders_dicts = DataLoaderKwargs(train_kwargs = {"batch_size": 32},
                                     val_kwargs = {"batch_size": 4},
                                     test_kwargs = {"batch_size": 3})

og = OrbitsGenerator(num_orbits_per_class=10,
                     homology_dimensions = homology_dimensions,
                     validation_percentage=0.2,
                     test_percentage=0.2)

dl_train, dl_val, dl_test = og.get_dataloader_orbits(dataloaders_dicts)

# %%

model = SetTransformer(
            dim_input=len(homology_dimensions),
            num_outputs=5,
            attention_type="induced_attention").double()

# %%
for x, y in dl_test:
    print(model(x).shape)
# %%
