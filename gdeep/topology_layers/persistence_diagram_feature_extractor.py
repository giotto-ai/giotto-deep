from typing import List, Union, Optional, Tuple

import numpy as np
import torch
from transformers import BatchFeature
from transformers.feature_extraction_sequence_utils import (
    FeatureExtractionMixin
)
# TODO: Use Explicit enum for return_tensors argument such that it is clear that
# the user is not allowed to pass an invalid value and the type checker will
# catch it.
# class TensorReturnType(ExplicitEnum):
#    PYTORCH = "pt"
#    NUMPY = "np"
#    TENSORFLOW = "tf"
#    TORCH = "torch"



class PersistenceDiagramFeatureExtractor(FeatureExtractionMixin):
    """Feature extractor for persistence diagrams.

    The feature extractor can be saved to a file and loaded from a file using
    the `save_pretrained` and `from_pretrained` methods.
    
    Examples::
        import numpy as np
    
        from gdeep.topology_layers import PersistenceDiagramFeatureExtractor
        from gdeep.utility.constants import DEFAULT_DATA_DIR

        # Create persistence diagram with one homology dimension.
        persistence_diagrams = np.random.rand(2, 10, 2)

        mean = np.array([[0.5, 0.5]])
        std = np.array([[0.1, 0.1]])

        pd_extractor = PersistenceDiagramFeatureExtractor(
            mean=mean,
            std=std,
            number_of_homology_dimensions=1,
            number_of_most_persistent_features=3,
        )

        # save the extractor to a file
        pd_extractor.save_pretrained('.')
        
        del pd_extractor
        pd_extractor = PersistenceDiagramFeatureExtractor.\
            from_pretrained('preprocessor_config.json')
                
        features = pd_extractor(persistence_diagrams)
        
        input_values = features['input_values']
        attention_masks = features['attention_mask']
    """
    mean: np.ndarray
    std: np.ndarray
    number_of_homology_dimensions: int
    number_of_most_persistent_features: Optional[int]
    
    def __init__(self,
                 number_of_homology_dimensions: int,
                 mean: Union[np.ndarray, List[List[float]], None] = None,
                 std: Union[np.ndarray,  List[List[float]], None] = None,
                 number_of_most_persistent_features: Optional[int] = None,
                 **kwargs,
                 ):
        assert number_of_homology_dimensions > 0, \
            "The number of most persistent features must be greater than 0."
        self.number_of_homology_dimensions = number_of_homology_dimensions
        if mean is None:
            self.mean = np.array([[0.0, 0.0]] * self.number_of_homology_dimensions)
        elif isinstance(mean, np.ndarray):
            self.mean = mean
        else:
            self.mean = np.array(mean)
        if std is None:
            self.std = np.array([[1.0, 1.0]] * self.number_of_homology_dimensions)
        elif isinstance(std, np.ndarray):
            self.std = std
        else:
            self.std = np.array(std)
        assert self.mean.shape == (self.number_of_homology_dimensions, 2), \
            "The mean must be a 2-dimensional array."
        assert self.std.shape == (self.number_of_homology_dimensions, 2), \
            "The std must be a 2-dimensional array."
        self.number_of_most_persistent_features = \
            number_of_most_persistent_features
        super().__init__(**kwargs)
    
    def __call__(self,
                 raw_persistence_diagrams: Union[np.ndarray, torch.Tensor,
                                                 List[np.ndarray],
                                                 List[torch.Tensor]],
                 padding_length: Optional[int] = None,
                 return_tensors: Optional[str] = 'np',
                 ) -> BatchFeature:
        """Return the processed features.
        
        Args:
            raw_persistence_diagrams: Either a single persistence diagram or a
            list of raw persistence diagrams.
            padding_length: The length of the padding.
            return_tensors: The return type. Either 'np' or 'pt'.
        """
        assert return_tensors in ['np', 'pt'], \
            "The return_tensors must be either 'np' or 'pt'.\n" \
            "Tensorflow is not supported yet."
        
        list_persistence_diagrams: List[np.ndarray] = []
        if isinstance(raw_persistence_diagrams, np.ndarray):
            if(raw_persistence_diagrams.ndim == 2):
                list_persistence_diagrams = [raw_persistence_diagrams]
            elif(raw_persistence_diagrams.ndim == 3):
                list_persistence_diagrams = raw_persistence_diagrams.tolist()
            else:
                raise ValueError("The persistence diagrams must be 2- or 3-dimensional.")
        elif isinstance(raw_persistence_diagrams, torch.Tensor):
            if(raw_persistence_diagrams.ndim == 2):
                list_persistence_diagrams = [raw_persistence_diagrams.detach().cpu()\
                    .numpy()]
            elif(raw_persistence_diagrams.ndim == 3):
                list_persistence_diagrams = raw_persistence_diagrams.detach().cpu()\
                    .numpy().tolist()
            else:
                raise ValueError("The persistence diagrams must be 2- or 3-dimensional.")
        else:
            for x in raw_persistence_diagrams:
                assert x.ndim == 2, "The persistence diagrams must be 2-dimensional."
                if isinstance(x, torch.Tensor):
                    list_persistence_diagrams.append(
                        x.detach().cpu().numpy()   
                    )
                else:
                    list_persistence_diagrams.append(x)
                    
                    
        # Pad the persistence diagrams.
        raw_persistence_diagrams, attention_masks = \
            self._pad_persistence_diagrams(
                list_persistence_diagrams, padding_length
            )
            
        # Filter k-most persistent features.
        if self.number_of_most_persistent_features is not None:
            raw_persistence_diagrams, attention_masks = \
                self._get_most_persistent_features(
                    raw_persistence_diagrams,
                    attention_masks,
                    self.number_of_most_persistent_features
                )
        
        # TODO : Filter the persistence diagrams by treshold.
        
        # Normalize the persistence diagrams.
        normalized_persistence_diagrams = self._normalize_persistence_diagrams(
            raw_persistence_diagrams
        )
        return_values = {'input_values': normalized_persistence_diagrams,
                        'attention_mask': attention_masks}
        return BatchFeature(return_values, return_tensors)

    def _normalize_persistence_diagrams(self,
                                        persistence_diagrams: np.ndarray,
    ) -> np.ndarray:
        """Normalize the first tow coordinates of persistence diagrams per
        homology dimension.

        Args:
            persistence_diagrams (np.ndarray):  The persistence diagrams of the
            shape (batch_size, number_of_points_per_diagram, 
            2 + len(homology_dimensions)). The first two coordinates are the
            coordinates of the points. The last coordinates are the one-hot
            encoded homology dimensions.

        Returns:
            np.ndarray: The normalized persistence diagrams.
        """
        if self.number_of_homology_dimensions == 1:
            assert persistence_diagrams.shape[-1] == 2, \
                "The persistence diagrams must have 2 dimensions in the last " \
                "dimension."
            persistence_diagrams[:, :, :2] -= self.mean[0]
            persistence_diagrams[:, :, :2] /= self.std[0]
            return persistence_diagrams
            
        assert persistence_diagrams.shape[-1] == 2 + self.number_of_homology_dimensions, \
            "The persistence diagrams must have the same number of " \
            "dimensions as the homology dimensions + 2."
        # Filter all tensors that have a one-hot encoded homology dimension
        # at position 2 + i, where i is the index of the homology dimension.
        for i in range(self.number_of_homology_dimensions):
            persistence_diagrams[persistence_diagrams[:, :, 2 + i] == 1
                ] -= np.concatenate((self.mean[i], np.array([0.0] * (self.number_of_homology_dimensions))))
            persistence_diagrams[persistence_diagrams[:, :, 2 + i] == 1
                ] /=  np.concatenate((self.std[i], np.array([1.0] * (self.number_of_homology_dimensions))))
        return persistence_diagrams
    
    def _get_most_persistent_features(self,
                                      persistence_diagrams: np.ndarray,
                                      attention_mask: np.ndarray,
                                      number_of_most_persistent_features: int
                                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the most persistent features of the persistence diagrams.
        
        Args:
            persistence_diagrams: The persistence diagrams of shape (batch_size,
             number_of_point_per_diagram, number_of_homology_dimensions + 2,)
            number_of_most_persistent_features: The number of most persistent
            features to keep.
            
        Returns:
            The most persistent features of the persistence diagrams together
            with the attention masks.
        """
        assert number_of_most_persistent_features > 0, \
            "The number of most persistent features must be greater than 0."
        assert number_of_most_persistent_features < persistence_diagrams.shape[1], \
            "The number of most persistent features must be smaller than the " \
            "number of persistence diagrams."
        
        args = np.argpartition(persistence_diagrams[:, :, 1] \
            - persistence_diagrams[:, :, 0],
                        -number_of_most_persistent_features, axis=1)\
                            [:, -number_of_most_persistent_features:]
        return (np.stack([persistence_diagrams[i, args[i]] 
                          for i in range(len(args))], axis=0),
                np.stack([attention_mask[i, args[i]] 
                          for i in range(len(args))], axis=0))
        
    def _pad_persistence_diagrams(self,
                                  raw_persistence_diagrams: List[np.ndarray],
                                  padding_length: Optional[int] = None,
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pad the persistence diagrams to the same length.
        
        Args:
            raw_persistence_diagrams: List of persistence diagrams.
            padding_length: Length to pad the persistence diagrams to.
            
        Returns:
            Tuple of padded persistence diagrams and mask.
        """
        if padding_length is None:
            padding_length = max([len(x) for x in raw_persistence_diagrams])
        padded_persistence_diagrams: List[np.ndarray] = []
        attention_mask: List[np.ndarray] = []
        for persistence_diagram in raw_persistence_diagrams:
            padded_persistence_diagrams.append(
                np.pad(persistence_diagram,
                          ((0, padding_length - len(persistence_diagram)), (0, 0)),
                            mode='constant',
                            constant_values=0)
            )
            attention_mask.append(
                np.pad(np.ones(len(persistence_diagram)),
                            (0, padding_length - len(persistence_diagram)),
                            mode='constant',
                            constant_values=0)
            )
            
        return (np.stack(padded_persistence_diagrams, axis=0), 
                    np.stack(attention_mask, axis=0))
    
    def __repr__(self) -> str:
        return f"PersistenceDiagramFeatureExtractor(mean={self.mean}, " \
                f"std={self.std}, " \
                f"homology_dimensions={self.number_of_homology_dimensions}, " \
                f"number_of_most_persistent_features={self.number_of_most_persistent_features})"
        
