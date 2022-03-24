# %% 
from IPython import get_ipython  # type: ignore
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from os.path import join
from gdeep.search import PersformerHyperparameterSearch

# %%
# This is how you use the api to search for the best hyperparameters for the MutagDataset 
# using the PersformerHyperparameterSearch class. The search is performed using the hyperparameter
# search space described in hpo_space file provided. Please customize the file to your own dataset.
# The results are written to the path_writer directory.
# Important: To be able to access the google storage buckets please set up an api authentication key on your machine.

dataset_name="MutagDataset"  # name of the dataset - has to exist in the datacloud buckets
download_directory = join("data", "DatasetCloud")  # directory where the dataset is downloaded
path_hpo_metadata = join('hpo_space', 'Mutag_hyperparameter_space.json')  # file describing the hyperparameter search space
path_writer = join("run", "auto_ml")  # directory where the runs are stored using the tensorboard writer

# Initialize the search object with the search parameters.
hpo = PersformerHyperparameterSearch(dataset_name=dataset_name,
                               download_directory=download_directory,
                               path_hpo_metadata=path_hpo_metadata,
                               path_writer=path_writer)

# Start the hyperparameter search.
hpo.search()