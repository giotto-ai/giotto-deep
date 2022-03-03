# %% 
from IPython import get_ipython  # type: ignore
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from os.path import join
from gdeep.topology_layers import HyperparameterSearch
from gdeep.search import GiottoSummaryWriter

#%%
dataset = "MutagDataset"
hyperparameter_space = join("hyperparameter_space", "mutag.json")

# Initialize the Tensorflow writer
writer = GiottoSummaryWriter(join("runs","mutag_persformer_hpo"))

HyperparameterSearch(dataset=dataset,
                     writer=writer,
                     hyperparameter_space=hyperparameter_space)