from typing import Tuple

import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import eigh

from gdeep.extended_persistence.gudhi_implementation import \
    graph_extended_persistence_gudhi
from gdeep.topology_layers.preprocessing import \
    _convert_single_graph_extended_persistence_to_one_hot_array

# Compute the heat kernel signature of a graph.
def _heat_kernel_signature(adj_mat: np.ndarray,
                          diffusion_parameter=1.0) -> np.ndarray:
    """Given a graph adjacency matrix, compute the heat kernel signature.
    
    Args:
        adj_mat (np.ndarray):
            The adjacency matrix of the graph.
        diffusion_parameter (float):
            The diffusion parameter of the heat kernel.
            
    Returns:
        np.ndarray:
            The heat kernel signature of the graph. The shape of the array is
            (num_vertices).
    """
    eigenvals, eigenvectors = _get_eigenvalues_eigenvectors(adj_mat)
    hks = (np.square(eigenvectors) * np.exp(-diffusion_parameter * eigenvals))\
        .sum(axis=1)
    return hks
    
def _get_eigenvalues_eigenvectors(adj_mat) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the eigenvector and eigenvalue of the adjacency matrix.
    
    Args:
        adj_mat (np.ndarray):
            The adjacency matrix of the graph.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            The eigenvectors and eigenvalues of the adjacency matrix.
    """
    # Compute the Laplacian matrix
    laplacian_mat = csgraph.laplacian(adj_mat, normed=True)
    # Compute the eigenvalues and eigenvectors of the Laplacian matrix
    eigenvalues, eigenvectors = eigh(laplacian_mat)
    return eigenvalues, eigenvectors

def graph_extended_persistence_hks(adj_mat: np.ndarray,
                                   diffusion_parameter: float = 1.0) \
                                       -> np.ndarray:
    """Compute the extended persistence of a graph.
    
    Args:
        adj_mat (np.ndarray):
            The adjacency matrix of the graph.
        diffusion_parameter (float):
            The diffusion parameter of the heat kernel.
            
    Returns:
        np.ndarray:
            The extended persistence of the graph.
    """
    # Compute the heat kernel signature
    hks = _heat_kernel_signature(adj_mat, diffusion_parameter)
    # Compute the extended persistence
    persistence_diagram = graph_extended_persistence_gudhi(adj_mat, hks)
    return _convert_single_graph_extended_persistence_to_one_hot_array(
        persistence_diagram
    )