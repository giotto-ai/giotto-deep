# %% [markdown]
# ## Benchmarking PersFormer on the graph datasets.
# We will compare the accuracy on the graph datasets of our SetTransformer
# based on PersFormer with the perslayer introduced in the paper:
# https://arxiv.org/abs/1904.09378

# %% [markdown]
# ## Benchmarking MUTAG
# We will compare the test accuracies of PersLay and PersFormer on the MUTAG
# dataset. It consists of 188 graphs categorised into two classes.
# We will train the PersFormer on the same input features as PersFormer to
# get a fair comparison.
# The features PersLay is trained on are the extended persistence diagrams of
# the vertices of the graph filtered by the heat kernel signature (HKS)
# at time t=10.
# The maximum (wrt to the architecture and the hyperparameters) mean test
# accuracy of PersLay is 89.8(±0.9) and the train accuracy with the same
# model and the same hyperparameters is 92.3.
# They performed 10-fold evaluation, i.e. splitting the dataset into
# 10 equally-sized folds and then record the test accuracy of the i-th
# fold and training the model on the 9 other folds.

# %%
# Import libraries:
from typing import Tuple
import numpy as np  # typing: ignore
import torch
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange  # typing: ignore
from os.path import join, isfile
from gdeep.topology_layers import ISAB, PMA

# %%
# Load extended persistence diagrams and additional features


def load_data(
        dataset: str = "MUTAG",
        path_dataset: str = "",
        filtrations=None,
        verbose: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """[summary]

    Args:
        dataset (str, optional): File name of the dataset to load. There should
            be a hdf5 file for the extended persistence diagrams of the dataset
            as well as a csv file for the additional features in the path
            dataset directory. Defaults
            to "MUTAG".
        path_dataset (str, optional): Directory name of the dataset to load.
            Defaults to None.
        filtrations ([type], optional): [description]. Defaults to None.
        verbose (bool, optional): If `True` print size of the loaded dataset.
            Defaults to False.
    """
    try:
        assert(isfile(join(path_dataset, dataset + ".hdf5")))
    except AssertionError:
        print(dataset + '.hdf5 not found in given directory!')