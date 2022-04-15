from typing import Tuple

import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import eigh

# Compute the heat kernel signature of a graph.
def heat_kernel_signature(adj_mat, diffusion_parameter=1.0) -> np.ndarray:
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
    eigenvals, eigenvectors = _get_eigenvector_eigenvalue(adj_mat)
    hks = np.square(eigenvectors) * np.exp(-diffusion_parameter * eigenvals)\
        .reshape(1, -1)
    return hks.sum()
    
def _get_eigenvector_eigenvalue(adj_mat) -> Tuple[np.ndarray, np.ndarray]:
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
    eigenvals, eigenvectors = eigh(laplacian_mat)
    return eigenvals, eigenvectors