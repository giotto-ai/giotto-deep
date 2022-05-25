# %%
import os
import numpy as np
import torch

from gdeep.data.persistence_diagrams import (
    OneHotEncodedPersistenceDiagram,
)
from gdeep.utility import DEFAULT_GRAPH_DIR

from gdeep.utility.utils import autoreload_if_notebook
autoreload_if_notebook()

# %%
file_path: str = os.path.join(DEFAULT_GRAPH_DIR, "MUTAG_10.1_extended_persistence", "diagrams")
pd = OneHotEncodedPersistenceDiagram.load(os.path.join(file_path, "graph_1_persistence_diagram.npy"))
names = ["Ord0", "Ext0", "Rel1", "Ext1"]
pd.plot(names)
# %%
