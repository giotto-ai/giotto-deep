import os
from typing import List, Tuple, Callable

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
                 transform: Callable = None
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
        self.transform = transform
        
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the persistence diagram and the corresponding label of the
        specified graph.
        
        Args:
            idx: The index of the graph in the dataset.
        
        Returns:
            The persistence diagram and the corresponding label of the specified
            graph.
        """
        # Load the persistence diagram
        persistence_diagram = np.load(
            os.path.join(self.output_dir, "diagrams",
                         f"graph_{idx}_persistence_diagram.npy")
            )
        
        # Load the label
        label = self.labels[idx][1]
        
        # Convert the persistence diagram to a tensor
        persistence_diagram_tensor = torch.tensor(
            persistence_diagram,
            dtype=torch.float32
            )
        
        # Convert the label to a tensor
        label_tensor = torch.tensor(
            label,
            dtype=torch.long
            )
        
        # Apply the transformation
        if self.transform is not None:
            persistence_diagram_tensor = self.transform(
                persistence_diagram_tensor
                )
        
        return persistence_diagram_tensor, label_tensor
         
        
    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]])\
                                                  -> Tuple[torch.Tensor,
                                                           torch.Tensor, 
                                                           torch.Tensor]:
        """
        Collate a batch of persistence diagrams and labels by padding the
        persistence diagrams to the same length.
        
        Args:
            batch: A list of tuples of the form (persistence_diagram, label).
            
        Returns:
            persistence_diagrams: A tensor of the form (batch_size, 
                                   persistence_diagram_length).
            masks: A tensor of the form (batch_size, 
                   persistence_diagram_length).
            labels: A tensor of the form (batch_size, 1).
        """
        # Get the lengths of the persistence diagrams
        lengths = torch.tensor([len(persistence_diagram) for 
                                persistence_diagram, _ in batch])
        
        # Pad the persistence diagrams to the maximum length of the batch
        max_length = int(lengths.max().item())
        persistence_diagrams = torch.zeros(
            len(batch), max_length, dtype=torch.float32
            )
        masks = torch.zeros(
            len(batch), max_length, dtype=torch.float32
            )
        
        for idx, (persistence_diagram, _) in enumerate(batch):
            persistence_diagrams[idx, :len(persistence_diagram)] = \
                persistence_diagram
            masks[idx, :len(persistence_diagram)] = 1.0
            
        # Convert the labels to a tensor
        labels = torch.tensor([label.item() for _, label in batch], 
                              dtype=torch.long
                              )
        
        return persistence_diagrams, masks, labels
    
