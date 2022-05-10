import os
from typing import List, Tuple, Callable, Union, Optional
from types import FunctionType

import numpy as np
import pandas as pd
from requests import HTTPError
import torch
from torch_geometric.datasets import TUDataset  # type: ignore
from torch_geometric.utils import to_dense_adj  # type: ignore
from tqdm import tqdm
from torch.utils.data import Dataset

from gdeep.extended_persistence.heat_kernel_signature import \
    graph_extended_persistence_hks
from gdeep.utility.constants import DEFAULT_GRAPH_DIR
from gdeep.utility._typing_utils import torch_transform


Tensor = torch.Tensor

class PersistenceDiagramFromGraphDataset(Dataset):
    """
    This class is used to load the persistence diagrams of the graphs in a
    dataset. All graph datasets in the TUDataset class are supported.
    """
    transform: Union[Callable[[Tensor], Tensor], None]
    
    def __init__(self,
                 dataset_name: str,
                 diffusion_parameter: float,
                 root: str = DEFAULT_GRAPH_DIR,
                 transform: Union[Callable[[torch.Tensor], torch.Tensor],
                                  Callable[[np.ndarray], np.ndarray],
                                  None] = None
                 ):
        """
        Initialize the dataset.
        
        Args:
            dataset_name: The name of the graph dataset to load, e.g. "MUTAG".
            diffusion_parameter: The diffusion parameter of the heat kernel
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
        
        # Check if transform is a function mapping a numpy array to a numpy array
        # if so, transform it to a callable that takes a tensor and returns a
        # tensor
        if transform is not None:
            self.transform = torch_transform(transform)
        else:
            self.transform = None
        
        # Check if the dataset exists in the specified directory
        if not os.path.exists(self.output_dir):
            print(f"Dataset {dataset_name} does not exist!")
            self._preprocess()
            
        # Load the labels
        self.labels: pd.DataFrame = pd.read_csv(
            os.path.join(self.output_dir, "labels.csv"),
            header=None,
            names=["graph_id", "label"]
            )
        
    def __repr__(self) -> str:
        """
        Return a string representation of the dataset.
        
        Returns:
            A string representation of the dataset.
        """
        return f"{self.__class__.__name__}(dataset_name={self.dataset_name}, " \
                f"diffusion_parameter={self.diffusion_parameter}, " \
                f"root={self.root}, transform={self.transform})"


    def _preprocess(self) -> None:
        """
        Preprocess the dataset and save the persistence diagrams and the labels
        in the output directory.
        The persistence diagrams are computed using the heat kernel signature
        method and then each diagram is saved in a separate npy file in the
        diagrams subdirectory of the output directory.
        The labels are saved in a csv file in the output directory.
        
        Args:
            output_dir: The directory where to save the persistence diagrams.
        """        
        # Load the dataset
        try:
            self.graph_dataset = TUDataset(root=DEFAULT_GRAPH_DIR,
                                        name=self.dataset_name,
                                        use_node_attr=False,
                                        use_edge_attr=False)
        except HTTPError as e:
             if e.code == 404:
                raise ValueError(f"Dataset {self.dataset_name} does not exist!")
             else:
                raise e
        
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
        for graph_idx, graph in tqdm(enumerate(self.graph_dataset),
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
            sorted_diagram = PersistenceDiagramFromGraphDataset.\
                _sort_diagram_by_lifetime(
                persistence_diagram_one_hot
                )
            
            # Save the persistence diagram in a file
            np.save(
                (os.path.join(self.output_dir, "diagrams",
                              f"graph_{graph_idx}_persistence_diagram.npy")),
                sorted_diagram
                )
            
            # Save the label
            labels.append((graph_idx, graph.y))

        # Save the labels in a csv file
        pd.DataFrame(labels, columns=["graph_idx", "label"]).to_csv(
            os.path.join(self.output_dir, "labels.csv"),
            index=False
            )

    @staticmethod
    def _sort_diagram_by_lifetime(diagram: np.ndarray) -> np.ndarray:
        filtered_diagram: np.ndarray = \
            diagram[
                    (diagram[:, 1] -
                        diagram[:, 0]).argsort()
                ]
        return filtered_diagram
        
    def __len__(self) -> int:
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
        label: int = int(self.labels.loc[idx, "label"])
        
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
                                   persistence_diagram_length, 6).
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
            len(batch), max_length, 2 + self.num_homology_types,
            dtype=torch.float32
            )
        masks = torch.zeros(
            len(batch), max_length, dtype=torch.float32
            )
        
        for idx, (persistence_diagram, _) in enumerate(batch):
            length: int = int(lengths[idx].item())
            persistence_diagrams[idx, :length] = \
                persistence_diagram[:length]
            masks[idx, :length] = 1.0
            
        # Convert the labels to a tensor
        labels = torch.tensor([label for _, label in batch],
                                dtype=torch.long
                                )
        
        return persistence_diagrams, masks, labels