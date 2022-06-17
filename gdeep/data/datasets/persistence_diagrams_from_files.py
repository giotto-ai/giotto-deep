from typing import Dict, Optional, Tuple, List
import os
from os.path import join

import torch
from torch.utils.data import DataLoader, Dataset

from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import OneHotEncodedPersistenceDiagram

class PersistenceDiagramFromFiles(Dataset[Tuple[OneHotEncodedPersistenceDiagram, int]]):
    file_path: str
    lenght: int
    labels: Dict[int, int]
    
    def __init__(self, 
                 file_path: str,
                 ):
        """
        Args:
            file_path: 
                The path to the persistence diagrams.
                num_homology_types: The number of homology types.
                transform: A function that transforms the data.
        """
        self.file_path = file_path
        self.len = len(os.listdir(join(self.file_path, 'diagrams')))
        self._load_labels()
        
        
    def __len__(self) -> int:
        """
        Return the length of the dataset.
        
        Returns:
            The length of the dataset.
        """
        return self.len
    
    def _load_labels(self) -> None:
        """
        Load the labels of the dataset.
        
        Returns:
            The labels of the dataset.
        """
        self.labels: Dict[int, int] = {}
        # load labels.csv with columns=["graph_idx", "label"] skipping the header
        with open(join(self.file_path, 'labels.csv')) as f:
            for line in f:
                if line.startswith('graph_idx'):
                    continue
                graph_idx, label = line.strip().split(',')
                self.labels[int(graph_idx)] = int(label)
        self.labels
    
    def __getitem__(self, index: int) -> Tuple[OneHotEncodedPersistenceDiagram, int]:
        """
        Return the item at the specified index.
        
        Args:
            index: 
                The index of the item.
            
        Returns:
            The item at the specified index.
        """
        diagram = OneHotEncodedPersistenceDiagram.load(join(self.file_path, 'diagrams',
                                                            f'{index}.npy'))
        label = self.labels[index]
        return diagram, label
    
    
def collate_fn_persistence_diagrams(self, batch: List[Tuple[torch.Tensor, torch.Tensor]])\
                                                -> Tuple[torch.Tensor,
                                                        torch.Tensor, 
                                                        torch.Tensor]:
    """
    Collate a batch of persistence diagrams and labels by padding the
    persistence diagrams to the same length.
    
    Args:
        batch: 
            A list of tuples of the form (persistence_diagram, label).
        
    Returns:
        persistence_diagrams: 
            A tensor of the form (batch_size, persistence_diagram_length, 6).
        masks: 
            A tensor of the form (batch_size, persistence_diagram_length).
        labels: 
            A tensor of the form (batch_size, 1).
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
