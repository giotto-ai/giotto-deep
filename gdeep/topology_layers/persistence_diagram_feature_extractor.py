from typing import List, Optional, Tuple, Union, Any, TypeVar

import numpy as np
import torch
from transformers import BatchFeature
from transformers.feature_extraction_sequence_utils import \
    FeatureExtractionMixin

# TODO: Use Explicit enum for return_tensors argument such that it is clear that
# the user is not allowed to pass an invalid value and the type checker will
# catch it.
# class TensorReturnType(ExplicitEnum):
#    PYTORCH = "pt"
#    NUMPY = "np"
#    TENSORFLOW = "tf"
#    TORCH = "torch"

array2d = np.ndarray[Any, Any]
array3d = np.ndarray[Any, Any]  #TODO: Add type annotation for 3d array

MultiplePersistenceDiagrams = Union[np.ndarray, torch.Tensor, 
                                    List[np.ndarray], List[torch.Tensor]]

#TODO: Add MinMaxScaler
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
    mean: array2d
    std: array2d
    number_of_homology_dimensions: int
    number_of_most_persistent_features: Optional[int]
    threshold: Optional[float]
    
    def __init__(self,
                 number_of_homology_dimensions: int,
                 mean: Union[array2d, List[List[float]], None] = None,
                 std: Union[array2d,  List[List[float]], None] = None,
                 number_of_most_persistent_features: Optional[int] = None,
                 treshold: Optional[float] = None,
                 **kwargs,
                 ):
        assert number_of_homology_dimensions > 0, \
            "The number of most persistent features must be greater than 0."
        self.number_of_homology_dimensions = number_of_homology_dimensions
        self._set_normalizing_parameters(mean, std)
        assert self.mean.shape == (self.number_of_homology_dimensions, 2), \
            "The mean must be a 2-dimensional array."
        assert self.std.shape == (self.number_of_homology_dimensions, 2), \
            "The std must be a 2-dimensional array."
        self.number_of_most_persistent_features = \
            number_of_most_persistent_features
        assert treshold is None or (0.0 <= treshold), \
            "The threshold must be between non-negative"
        self.treshold = treshold
        
        super().__init__(**kwargs)

    def _set_normalizing_parameters(self, mean: array2d,
                                    std: array2d) -> None:
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
    
    def __call__(self,
                 raw_persistence_diagrams: MultiplePersistenceDiagrams,
                 padding_length: Optional[int] = None,
                 return_tensors: Optional[str] = 'np',
                 ) -> BatchFeature:
        """Return the processed features.
        
        Args:
            raw_persistence_diagrams: Either a single persistence diagram or a
            list of raw persistence diagrams.
            padding_length: The length of the padding.
            return_tensors: The return type. Either 'np' or 'pt'.
            
        Returns:
            BatchFeature: The processed features in the following format:
            {'input_values': np.ndarray, 'attention_mask': np.ndarray}
            where the input_values is a 3-dimensional array of shape
            (batch_size, number_of_points_in_persistence_diagram,
            2 + number_of_homology_dimensions)
            and the attention_mask is a 2-dimensional array of shape
            (batch_size, number_of_points_in_persistence_diagram).
        """
        assert return_tensors in ['np', 'pt'], \
            "The return_tensors must be either 'np' or 'pt'.\n" \
            "Tensorflow is not supported yet."
        
        persistence_diagrams_list = \
            self._get_persistence_diagrams_list(raw_persistence_diagrams)
            
        # Check that the list_persistence_diagrams is a list of persistence
        # diagrams with the correct number of homology dimensions.
        assert self._check_persistence_diagrams(persistence_diagrams_list), \
            "The persistence diagrams are not valid."
            
        # Filter the persistence diagrams by treshold.
        if self.treshold is not None:
            persistence_diagrams_list = \
                [PersistenceDiagramFeatureExtractor._filter_by_treshold(
                    persistence_diagram, self.treshold
                ) for persistence_diagram in persistence_diagrams_list]
            
        persistence_diagrams: array3d
        # Pad the persistence diagrams.
        if len(persistence_diagrams_list) > 1 or \
            padding_length is not None:
            persistence_diagrams, attention_masks = \
                self._pad_persistence_diagrams(
                    persistence_diagrams_list, padding_length
                )
        else:
            # If there is only one persistence diagram and there is no padding.
            # -> Transform the persistence diagrams to a 3-dimensional array.
            persistence_diagrams = np.array(persistence_diagrams_list)
            
        assert persistence_diagrams.ndim == 3, \
            "The persistence diagrams must be 3-dimensional."
            
        # Filter k-most persistent features.
        if self.number_of_most_persistent_features is not None:
            persistence_diagrams, attention_masks = \
                self._get_most_persistent_features(
                    persistence_diagrams,
                    attention_masks,
                    self.number_of_most_persistent_features
                )
        
        
        # Normalize the persistence diagrams.
        persistence_diagrams = self._normalize_persistence_diagrams(
            persistence_diagrams
        )
        return_values = {'input_values': persistence_diagrams,
                        'attention_mask': attention_masks}
        return BatchFeature(return_values, return_tensors)
    
    @staticmethod
    def _filter_by_treshold(persistence_diagram: array2d,
                            treshold: float) -> array2d:
        """Filter the persistence diagram by lifetime treshold.
           Note: If the birth time is greater than the death time,
              the lifetime is set to birth time - death time.
        Args:
            persistence_diagram: The persistence diagram.
            treshold: The treshold.
            
        Returns:
            array2d: The filtered persistence diagram.
        """
        absolute_lifetime = np.abs(persistence_diagram[:, 1] 
                                   - persistence_diagram[:, 0])
        return persistence_diagram[absolute_lifetime > treshold]
    
         

    def _get_persistence_diagrams_list(self, 
                                       raw_persistence_diagrams: \
                                           MultiplePersistenceDiagrams
        ) -> List[array2d]:
        list_persistence_diagrams: List[np.ndarray] = []
        if isinstance(raw_persistence_diagrams, np.ndarray):
            list_persistence_diagrams =\
                self.__class__.persistence_diagram_list_from_array(raw_persistence_diagrams)
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
        return list_persistence_diagrams

    @staticmethod
    def persistence_diagram_list_from_array(raw_persistence_diagrams: array3d) -> \
        List[array2d]:
        if(raw_persistence_diagrams.nide == 2):
            return [raw_persistence_diagrams]
        elif(raw_persistence_diagrams.ndim == 3):
            return raw_persistence_diagrams.tolist()
        else:
            raise ValueError("The persistence diagrams must be 2- or 3-dimensional.")

    def _check_persistence_diagrams(self,
                                    list_persistence_diagrams: List[np.ndarray],
                                    ) -> bool:
        """Check if the persistence diagrams all have correct number of
        homology dimensions.

        Args:
            list_persistence_diagrams (List[np.ndarray]): The persistence
            diagrams.

        Returns:
            bool: True if the persistence diagrams all have correct number of
            homology dimensions.
        """
        for persistence_diagram in list_persistence_diagrams:
            if persistence_diagram.ndim != 2:
                return False
            elif self.number_of_homology_dimensions == 1:
                return persistence_diagram.shape[-1] == 2
            if persistence_diagram.shape[-1] != \
                2 + self.number_of_homology_dimensions:
                return False
        return True

    # @staticmethod
    # def _get_number_of_persistence_diagrams(persistence_diagrams: 
    #     Union[np.ndarray, torch.Tensor,
    #           List[np.ndarray], List[torch.Tensor]]) -> int:
    #     """Return the number of persistence diagrams.
        
    #     Args:
    #         persistence_diagrams: Either an array of tensors of a single or 
    #         multiple persistence diagram or a list of persistence diagrams.
        
    #     Returns:
    #         The number of persistence diagrams.
    #     """
    #     if isinstance(persistence_diagrams, np.ndarray) or \
    #         isinstance(persistence_diagrams, torch.Tensor):
    #         if(persistence_diagrams.ndim == 2):
    #             return 1
    #         else:
    #             return persistence_diagrams.shape[0]
    #     else:
    #         return len(persistence_diagrams)
        
        
    def _normalize_persistence_diagrams(self,
                                        persistence_diagrams: \
                                        array3d,
    ) -> array3d:
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
                                      persistence_diagrams: array3d,
                                      attention_mask: array2d,
                                      number_of_most_persistent_features: int
                                      ) -> Tuple[array3d, \
                                          array2d]:
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
                                  raw_persistence_diagrams: List[np.ndarray[Any,
                                                                            Any]],
                                  padding_length: Optional[int] = None,
                                    ) -> Tuple[array3d, \
                                          array2d]:
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
