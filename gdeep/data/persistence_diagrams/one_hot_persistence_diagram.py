from typing import Any, Dict, Set, Tuple, TypeVar

import numpy as np
import torch

from gdeep.utility.utils import flatten_list_of_lists

T = TypeVar("T")
Tensor = torch.Tensor
Array = np.ndarray

class OneHotEncodedPersistenceDiagram(Tensor):
    """This class represents a single one-hot encoded persistence diagram.
    """
    def __init__(self, data: Tensor):  # type: ignore
        super().__init__()
        _check_if_valid(data)
        self.data = _sort_by_lifetime(data)
    
    def __add__(self, other: Any) -> Tensor:
        raise ValueError("The addition of persistence diagrams is not supported.")
    
    def __sub__(self, other: Any) -> Tensor:
        raise ValueError("The subtraction of persistence diagrams is not supported.")
    
    def __div__(self, other: Any) -> Tensor:
        return super().__div__(other)
    
    def get_num_homology_dimensions(self) -> int:
        """This method returns the number of homology dimensions.
        """
        return self.shape[-1] - 2
    
    def get_num_points(self) -> int:
        """This method returns the number of points.
        """
        return self.shape[0]
    
    def __repr__(self):
        return (f"OneHotEncodedPersistenceDiagram({self.shape})\n"
                f"{self.get_num_homology_dimensions()} homology dimensions\n"
                f"{self.shape[0]} points\ndata:\n {super().__repr__()}")

    def save(self, path: str) -> None:
        """This method saves the persistence diagram to a file.
        """
        torch.save(self, path)
        
    def get_all_points_in_homology_dimension(self, homology_dimension: int) -> Tensor:
        """This method returns all points in a given homology dimension.
        """
        assert homology_dimension < self.get_num_homology_dimensions(), \
        "The homology dimension must be smaller than the number of homology dimensions."
        return self[
            torch.where(
                self[:, 2 + homology_dimension] == 1.0
            )
        ]
        
    def plot(self) -> None:
        """This method plots the persistence diagram.
        """
        import matplotlib.pyplot as plt
        for t in range(self.get_num_homology_dimensions()):
            points = self.get_all_points_in_homology_dimension(t)
            plt.scatter(points[:, 0], points[:, 1], label=f"homology dimension {t}")
        plt.show()
        
    def get_lifetimes(self) -> Tensor:
        """This method returns the lifetimes of the points.
        """
        return torch.Tensor(self[:, 1]) - torch.Tensor(self[:, 0])
    
    def filter_by_lifetime(self, min_lifetime: float, max_lifetime: float) -> \
        'OneHotEncodedPersistenceDiagram':
        """This method filters the persistence diagram by lifetime.
        """
        lifetime: Tensor = self.get_lifetimes()
        return self[  # type: ignore
            torch.where(
                (lifetime >= min_lifetime) & (lifetime <= max_lifetime)
            )
        ]
        
    @staticmethod
    def from_numpy(data: Array) -> 'OneHotEncodedPersistenceDiagram':
        """This method creates a persistence diagram from a numpy array.
        """
        # assert data.dtype == np.float32, "The data must be of type np.float32. Otherwise,"\
        #     " the data will not be correctly converted."
        return OneHotEncodedPersistenceDiagram(torch.from_numpy(data.astype(np.float32)))
    

def _check_if_valid(data) -> None:
    if data.ndimension() != 2:
        raise ValueError("The input should be a 2-dimensional tensor."
                            f"The input has {data.ndimension()} dimensions.")
    assert data.shape[-1] > 2, \
        "The input should have at least one homology dimensions."
    assert torch.all(data[:, 2:] >= -1e-5) and \
        torch.allclose(data[:, 2:].sum(dim=1), torch.tensor(1.0)), \
            "The homology dimension should be one-hot encoded."

def _sort_by_lifetime(data: Tensor) -> Tensor:
    """This function sorts the points by their lifetime.
    """
    return data[(
        data[:, 1] - data[:, 0]
    ).argsort()]

def get_one_hot_encoded_persistence_diagram_from_gtda(persistence_diagram: Array) \
    -> OneHotEncodedPersistenceDiagram:
    """This function takes a single persistence diagram from giotto-tda and returns a one-hot encoded
    persistence diagram.
    
    Args:
        persistence_diagram: An array of shape (num_points, 3) where the first two columns
                                represent the coordinates of the points and the third column
                                represents the index of the homology dimension.
        
    Returns:
        A one-hot encoded persistence diagram. If the persistence diagram has only one homology
        dimension, the third column will be filled with ones.
    """
    assert persistence_diagram.ndim == 2 and persistence_diagram.shape[1] >=2, \
        "The input should be a 2-dimensional array of shape (num_points, 3) or (num_points, 2)."

    if persistence_diagram.shape[1] == 2:
        return OneHotEncodedPersistenceDiagram(
            torch.stack((torch.tensor(persistence_diagram),
                         torch.ones(persistence_diagram.shape[0])), dim=1))
    else:
        homology_types: Set[int] = set([int(i) for i in persistence_diagram[:, 2]])
        type_to_one_hot_encoding: Dict[int, int] = {
            i: j for j, i in enumerate(homology_types)
        }
        one_hot_encoding: Tensor = torch.zeros(persistence_diagram.shape[0],
                                                len(homology_types))
        # TODO: Fill the one-hot encoding in a vectorized manner.
        for i, j in enumerate(persistence_diagram[:, 2]):
            one_hot_encoding[i, type_to_one_hot_encoding[int(j)]] = 1
        birth_death_diagram: Tensor = torch.tensor(persistence_diagram[:, :2])
            
        return OneHotEncodedPersistenceDiagram(
            torch.stack((birth_death_diagram,
                         one_hot_encoding), dim=1))
        
def get_one_hot_encoded_persistence_diagram_from_gudhi_extended(
    diagram: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) \
        -> OneHotEncodedPersistenceDiagram:
    """Convert an extended persistence diagram of a single graph to an
    array with one-hot encoded homology type.
    Args:
        diagram (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
            The diagram of an extended persistence of a single graph.
    
    Returns:
        np.ndarray:
            The diagram in one-hot encoded homology type of size
            (num_points, 6).
    """
    # Get the length of each array
    lengths = [len(array) for array in diagram]
    
    if lengths == [0, 0, 0, 0]:
        return OneHotEncodedPersistenceDiagram(torch.zeros((0, 6)))
    
    # One-hot encode the homology type
    homology_type: np.ndarray = np.array(
        flatten_list_of_lists(
            [[i] * lengths[i] for i in range(4)]
        )
    )
    homology_type_one_hot = np.zeros((sum(lengths), 4))
    homology_type_one_hot[np.arange(sum(lengths)), homology_type] = 1
    
    # Concatenate the arrays
    diagram_one_hot = np.concatenate([sub_diagram for sub_diagram in diagram],
                                     axis=0)
    diagram_one_hot = np.concatenate([diagram_one_hot, homology_type_one_hot],
                                     axis=1)

    return OneHotEncodedPersistenceDiagram.from_numpy(diagram_one_hot)