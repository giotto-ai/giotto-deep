# %%
from typing import Callable, Union
import torch

from gdeep.data import PersistenceDiagramFromGraphDataset
from gdeep.utility import autoreload_if_notebook
autoreload_if_notebook()
# %%
graph_dataset_name = 'MUTAG_wtf'
persistence_dataset = PersistenceDiagramFromGraphDataset(
    graph_dataset_name,
    10.1,
)
# %%
