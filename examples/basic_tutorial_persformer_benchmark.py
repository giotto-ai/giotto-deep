# %% [markdown]
# #Basic tutorial: Training a topological model
#### Author: Raphael Reinauer
#### Date: 2022-04-05
# This short tutorial shows how to use the GDeep framework to train topological
# models using the topological datasets provided by the GDeep dataset cloud.

# The main steps of the tutorial are the following:
# 1. Specify the dataset you want to use.
# 2. Specify the model and the hyperparameter space you want to use.
# 3. Run a large scale hyperparameter search to find the good hyperparameters.

# %%
# This snippet will deactivate autoreload if this file
# is run as a script and activate it if it is run as a notebook.
from gdeep.utility.utils import autoreload_if_notebook

autoreload_if_notebook()
# Include necessary imports
from os.path import join

# Import the GDeep hpo module
from gdeep.search import PersformerHyperparameterSearch

# %% [markdown]
# ## Training a topological model with the Dataset Cloud
# In this tutorial we will use the our custom datasets storage
# on [Google Cloud Datastore](https://cloud.google.com/datastore/) to
# load datasets and train a topological model.
# The dataset cloud storage contain a variety of topological datasets
# that can be easily used in GDeep.
# We will use the Mutag dataset from the
# [Mutagenicity Benchmark](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5276825/)
# to do performance benchmarking for the Persformer model.
# For this benchmark we will use the GDeep
# [Persformer](https://doi.org/10.48550/arXiv.2112.15210) model,
# the GDeep pipeline and the GDeep hyperparameter search.
# With only a few lines of code we can train multiple topological models
# with different hyperparameters and evaluate the performance of the model.

# %%
# This is how you use the api to search for the best hyperparameters for
# the MutagDataset using the PersformerHyperparameterSearch class.
# The search is performed using the hyperparameter
# search space described in hpo_space file provided.
# Please customize the file to your own dataset.
# The results are written to the path_writer directory.

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
