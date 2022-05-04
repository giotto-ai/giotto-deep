from typing import NewType, List, Tuple, Callable, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

Tensor = torch.Tensor
Array = np.ndarray

def _create_preprocessing_transform(dataset: Dataset,
                                   test_indices: Optional[Array] = None,
                                   normalize_features: bool = True,
                                   num_points_to_keep: Optional[int] = None,
                                   ) -> Callable[[Tensor], Tensor]:
    
    """
    Create a preprocessing transform for the data in the specified dataloader.
    
    Args:
        dataloader(DataLoader): The dataloader to preprocess.
        test_indices(Optional[Array]): The indices of the test set. If None,
            the whole dataset is used.
        normalize_features(bool): Whether to normalize the features.
        num_points_to_keep(Optional[int]): The number of the most persistent
            points to keep.
        
    Returns:
        The preprocessing transform.
    """
    # Find the mean and standard deviation of the features
    if normalize_features:
        features_mean: float = torch.mean(
            torch.stack([x[:2] for x in dataset[test_indices]], dim=0), dim=0
        ).item()
        features_std: float = torch.std(
            torch.stack([x[:2] for x in dataset[test_indices]], dim=0), dim=0
        ).item()
    
    # Create the preprocessing transform
    def transform(x: Tensor) -> Tensor:
        """
        Transform the input data.
        
        Args:
            x: The input data.
        """
        # Normalize the features
        if normalize_features:
            x[:, :2] = (x[:, :2] - features_mean) / features_std
        
        return x
    
    return transform
    
def keep_most_persistent_points(x: Tensor, k: int) -> Tensor:
    """
    Keep the k-most persistent points in the persistence diagrams.
     
    Args:
        Tensor: The persistence diagrams.
        k: The number of the most persistent points to keep.
         
    Returns:
        The persistence diagrams with the k-most persistent points kept.
    """
    # Find the indices of the k-most persistent points
    indices = torch.argsort(x[1] - x[0])[-k:]
    
    # Return the k-most persistent points
    return x[indices]