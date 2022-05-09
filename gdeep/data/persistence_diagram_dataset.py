from dataclasses import dataclass
from typing import NewType, List, Tuple, Callable, Union, Optional, Type

import numpy as np
import torch
from torch.utils.data import Dataset


Tensor = torch.Tensor

@dataclass
class PersistenceDiagramDataset(Dataset):
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    stateful_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    
    
    def set_transform(self, 
                      transform: Union[Callable[[torch.Tensor],
                                                torch.Tensor], None]):
        if(self.transform is not None):
            self.transform = transform
        self.transform = transform
    
    
    def set_stateful_transform(
        self,
        transform: Callable[[torch.Tensor],
        torch.Tensor]) -> None:
        """
        Set the stateful transform for the dataset.
        
        Args:
            transform: The stateful transform.
        """
        if(self.stateful_transform is not None):
            raise ValueError("The stateful transform is already set.")
        self.stateful_transform = transform