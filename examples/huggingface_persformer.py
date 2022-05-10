# %%
from tkinter import N
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
from typing import NewType

# Create a subtype of tensor representing one-hot persistence diagrams with
# almost zero runtime overhead.
OneHotPersistenceDiagram = NewType('OneHotPersistenceDiagram', torch.Tensor)


def one_hot_persistence_diagram(persistence_diagram: torch.Tensor) \
    -> OneHotPersistenceDiagram:
    """Convert a persistence diagram to one-hot encoding."""
    assert persistence_diagram.ndim == 2
    # ..... additional assertions here ...
    
    return OneHotPersistenceDiagram(persistence_diagram)

x = torch.tensor([[0.0, 1.0, 0.0, 1.0],
                  [0.2, 0.4, 1.0, 0.0]])

pd = one_hot_persistence_diagram(x)

def get_number_of_homology_type(persistence_diagram: OneHotPersistenceDiagram) \
    -> int:
    """Get the number of homology types in a persistence diagram."""
    return persistence_diagram.shape[1] - 2

get_number_of_homology_type(pd) # = 2
get_number_of_homology_type(x)  # Will cause a TypeError

# Otherwise we can use pd as a usual pytorch tensor
one_hot_persistence_diagram(pd)
# %%
