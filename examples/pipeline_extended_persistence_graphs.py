
# %%
# Autoreload modules
from gdeep.utility.utils import autoreload_if_notebook
autoreload_if_notebook()
# %%
import os
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.datasets import TUDataset  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from torch_geometric.utils import to_dense_adj  # type: ignore

from gdeep.extended_persistence.heat_kernel_signature import \
    graph_extended_persistence_hks
from gdeep.utility.constants import DEFAULT_GRAPH_DIR
# %%

dataset_name = "REDDIT-BINARY"
diffusion_parameter: float = 10.0

# Load the dataset
graph_dataset = TUDataset(root=DEFAULT_GRAPH_DIR,
                 name=dataset_name,
                 use_node_attr=False,
                 use_edge_attr=False)
# %%

# Compute the heat kernel signature and the extended persistence of all the
# graphs in the dataset and save them in a file

# Define directory where to save the results
output_dir = os.path.join(DEFAULT_GRAPH_DIR, dataset_name + 
                          "_extended_persistence")
# Create the directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "diagrams"))
else:
    raise ValueError("Output directory already exists!")


labels: List[Tuple[int, int]] = []

for graph_idx, graph in enumerate(graph_dataset):
    if graph_idx % 100 == 0:
        print(f"Processing graph {graph_idx}")
    
    # Get the adjacency matrix
    adj_mat: np.ndarray = to_dense_adj(graph.edge_index)[0].numpy()
    
    # Compute the extended persistence
    persistence_diagram_one_hot = \
        graph_extended_persistence_hks(adj_mat, 
                                       diffusion_parameter=diffusion_parameter)
    

    # Save the persistence diagram in a file
    np.save(
        (os.path.join(output_dir, "diagrams",
                      f"graph_{graph_idx}_persistence_diagram.npy")),
        persistence_diagram_one_hot
        )
    labels.append((graph_idx, graph.y.item()))
    
# Save the labels in a csv file
np.savetxt(
    (os.path.join(output_dir, "labels.csv")),
    labels,
    delimiter=",",
    fmt="%d"
    )

# %%
