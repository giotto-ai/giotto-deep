import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import \
    OneHotEncodedPersistenceDiagram
from gdeep.utility.extended_persistence import graph_extended_persistence_hks
from gdeep.utility.constants import DEFAULT_GRAPH_DIR
from torch_geometric.datasets import TUDataset  # type: ignore
from torch_geometric.utils import to_dense_adj  # type: ignore
from tqdm import tqdm

PD = OneHotEncodedPersistenceDiagram


class PersistenceDiagramFromGraphBuilder:
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
            dataset_name:
                The name of the graph dataset to load, e.g. "MUTAG".
            diffusion_parameter:
                The diffusion parameter of the heat kernel
                signature. These are usually chosen to be as {0.1, 1.0, 10.0}.
        """
        self.dataset_name: str = dataset_name
        self.diffusion_parameter: float = diffusion_parameter
        self.num_homology_types: int = 4
        self.root: str = root
        self.output_dir: str = os.path.join(root,
                                       dataset_name + "_" + 
                                       str(diffusion_parameter) +
                                       "_extended_persistence")
        
    def create(self):
        # Check if the dataset exists in the specified directory
        if not os.path.exists(self.output_dir):
            print(f"Dataset {self.dataset_name} does not exist!")
            self._preprocess()
        else:
            print(f"Dataset {self.dataset_name} already exists!"
                " Skipping dataset will not be created.")
        
    def __repr__(self) -> str:
        """
        Return a string representation of the dataset.
        
        Returns:
            A string representation of the dataset.
        """
        return f"{self.__class__.__name__}(dataset_name={self.dataset_name}, " \
                f"diffusion_parameter={self.diffusion_parameter}, " \
                f"root={self.root})"


    def _preprocess(self) -> None:
        """
        Preprocess the dataset and save the persistence diagrams and the labels
        in the output directory.
        The persistence diagrams are computed using the heat kernel signature
        method and then each diagram is saved in a separate npy file in the
        diagrams subdirectory of the output directory.
        The labels are saved in a csv file in the output directory.
        
        Args:
            output_dir:
                The directory where to save the persistence diagrams.
        """        
        # Load the dataset
        self.graph_dataset = TUDataset(root=DEFAULT_GRAPH_DIR,
                                    name=self.dataset_name,
                                    use_node_attr=False,
                                    use_edge_attr=False)
    
        # Create the directory if it does not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, "diagrams"))
        else:
            # This should not be reached
            raise ValueError("Output directory already exists!")
            
        num_graphs = len(self.graph_dataset)
        
        labels: List[Tuple[int, int]] = []
        
        print("Computing the persistence diagrams...")
        for graph_idx, graph in tqdm(enumerate(self.graph_dataset),  # type: ignore
                                     total=num_graphs):
            
            # Get the adjacency matrix
            adj_mat: np.ndarray = to_dense_adj(graph.edge_index)[0].numpy()
            
            # Compute the extended persistence
            persistence_diagram_one_hot = \
                graph_extended_persistence_hks(adj_mat, 
                                               diffusion_parameter=
                                               self.diffusion_parameter)
            # Sort the diagram by the persistence lifetime, i.e. the second
            # column minus the first column
            
            # Save the persistence diagram in a file
            persistence_diagram_one_hot.save(
                os.path.join(self.output_dir, "diagrams",
                              f"{graph_idx}.npy")
            )
            
            # Save the label
            labels.append((graph_idx, graph.y.item()))

        # Save the labels in a csv file
        pd.DataFrame(labels, columns=["graph_idx", "label"]).to_csv(
            os.path.join(self.output_dir, "labels.csv"),
            index=False
            )
