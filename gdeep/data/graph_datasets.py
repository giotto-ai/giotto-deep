import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset  # type: ignore
from torch_geometric.utils import to_dense_adj  # type: ignore

from gdeep.extended_persistence.heat_kernel_signature import \
    graph_extended_persistence_hks
from gdeep.utility.constants import DEFAULT_GRAPH_DIR
# %%

class PersistenceDiagramFromGraphDataset(Dataset):
    """
    This class is used to load the persistence diagrams of the graphs in a
    dataset. All graph datasets in the TUDataset class are supported.
    """
    def __init__(self,
                 dataset_name: str,
                 diffusion_parameter: float,
                 root: str = DEFAULT_GRAPH_DIR,
                 ):
        """
        Initialize the dataset.
        
        Args:
            dataset_name: The name of the graph dataset to load, e.g. "MUTAG".
            diffusion_parameter: The diffusion parameter of the heat kernel
                signature. These are usually chosen to be as {0.1, 1.0, 10.0}.
        """
        self.dataset_name = dataset_name
        self.diffusion_parameter = diffusion_parameter
        self.root = root
        self.output_dir = os.path.join(root, dataset_name)
        
        # Check if the dataset exists in the specified directory
        if not os.path.exists(os.path.join(root, dataset_name)):
            self.preprocess()
            
        # Load the labels
        self.labels = np.loadtxt(
            os.path.join(self.output_dir, "labels.txt"),
            dtype=np.int32
            )
        
    def preprocess(self):
        """
        Preprocess the dataset and save the persistence diagrams in a file.
        
        Args:
            output_dir: The directory where to save the persistence diagrams.
        """
        # Create the directory if it does not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, "diagrams"))
        else:
            raise ValueError("Output directory already exists!")
        
        # Load the dataset
        self.graph_dataset = TUDataset(root=DEFAULT_GRAPH_DIR,
                                    name=self.dataset_name,
                                    use_node_attr=False,
                                    use_edge_attr=False)
        
        labels: List[Tuple[int, int]] = []
        
        for graph_idx, graph in enumerate(self.graph_dataset):
            if graph_idx % 100 == 0:
                print(f"Processing graph {graph_idx}")
            
            # Get the adjacency matrix
            adj_mat: np.ndarray = to_dense_adj(graph.edge_index)[0].numpy()
            
            # Compute the extended persistence
            persistence_diagram_one_hot = \
                graph_extended_persistence_hks(adj_mat, 
                                               diffusion_parameter=
                                               self.diffusion_parameter)
            
            # Save the persistence diagram in a file
            np.save(
                (os.path.join(self.output_dir, "diagrams",
                              f"graph_{graph_idx}_persistence_diagram.npy")),
                persistence_diagram_one_hot
                )
            
            labels.append((graph_idx, graph.y.item()))

        # Save the labels in a csv file
        np.savetxt(
            (os.path.join(self.output_dir, "labels.csv")),
            labels,
            delimiter=",",
            fmt="%d"
            )
        
    def __len__(self):
        """
        Return the number of persistence diagrams in the dataset.
        """
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Return the persistence diagram of the graph in the dataset at the
        specified index.
        
        Args:
            idx: The index of the graph in the dataset.
            
        Returns:
            The persistence diagram of the graph in the dataset at the specified
            index.
        """
        # Load the persistence diagrams
        return np.load(
            os.path.join(self.output_dir, "diagrams",
                         f"graph_{idx}_persistence_diagram.npy")
            ), self.labels[idx]
        
        
    def collate_fn(self, batch: List[np.ndarray]) -> Tuple[torch.Tensor, 
                                                           torch.Tensor]:
        """
        Collate the persistence diagrams of the graphs in the batch.
        
        Args:
            batch: The batch of persistence diagrams.
            
        Returns:
            The batch of persistence diagrams as a tensor.
        """
        # TODO: Implement this method
        pass
    
    
# dataset_name = "REDDIT-BINARY"
# diffusion_parameter: float = 10.0

# # Check if DEFAULT_GRAPH_DIR exists
# if not os.path.exists(DEFAULT_GRAPH_DIR):
#     raise ValueError("The directory {} does not exist!")

# # Load the dataset
# graph_dataset = TUDataset(root=DEFAULT_GRAPH_DIR,
#                  name=dataset_name,
#                  use_node_attr=False,
#                  use_edge_attr=False)

# # Compute the heat kernel signature and the extended persistence of all the
# # graphs in the dataset and save them in a file

# # Define directory where to save the results
# output_dir = os.path.join(DEFAULT_GRAPH_DIR, dataset_name + 
#                           "_extended_persistence")
# # Create the directory if it does not exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#     os.makedirs(os.path.join(output_dir, "diagrams"))
# else:
#     raise ValueError("Output directory already exists!")


# labels: List[Tuple[int, int]] = []

# for graph_idx, graph in enumerate(graph_dataset):
#     if graph_idx % 100 == 0:
#         print(f"Processing graph {graph_idx}")
    
#     # Get the adjacency matrix
#     adj_mat: np.ndarray = to_dense_adj(graph.edge_index)[0].numpy()
    
#     # Compute the extended persistence
#     persistence_diagram_one_hot = \
#         graph_extended_persistence_hks(adj_mat, 
#                                        diffusion_parameter=diffusion_parameter)
    

#     # Save the persistence diagram in a file
#     np.save(
#         (os.path.join(output_dir, "diagrams",
#                       f"graph_{graph_idx}_persistence_diagram.npy")),
#         persistence_diagram_one_hot
#         )
#     labels.append((graph_idx, graph.y.item()))
    
