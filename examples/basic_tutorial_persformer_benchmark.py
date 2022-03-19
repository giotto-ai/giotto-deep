# %% 
from IPython import get_ipython  # type: ignore
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from os.path import join
from gdeep.search import PersformerHyperparameterSearch

# %%
# This is how you use the api to search for the best hyperparameters for the MutagDataset 
# using the PersformerHyperparameterSearch class. The search is performed using the 
# hpo_space file provided. The results are written to the path_writer directory.

dataset_name="MutagDataset"
download_directory = join("data", "DatasetCloud")
path_hpo_metadata = join('hpo_space', 'Mutag_hyperparameter_space.json')
path_writer = join("run", "auto_ml")

hpo = PersformerHyperparameterSearch(dataset_name=dataset_name,
                               download_directory=download_directory,
                               path_hpo_metadata=path_hpo_metadata,
                               path_writer=path_writer)

hpo.search()