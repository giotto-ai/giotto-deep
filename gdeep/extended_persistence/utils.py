from typing import Tuple

import gudhi as gd  # type: ignore
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csgraph

from gdeep.utility.utils import flatten_list_of_lists
from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import (
    OneHotEncodedPersistenceDiagram,
    get_one_hot_encoded_persistence_diagram_from_gudhi_extended,
)

Array = np.ndarray

# Compute the heat kernel signature of a graph.
def _heat_kernel_signature(adj_mat: Array,
                          diffusion_parameter: float = 1.0) -> Array:
    """Given a graph adjacency matrix, compute the heat kernel signature.
    
    Args:
        adj_mat:
            The adjacency matrix of the graph.
        diffusion_parameter:
            The diffusion parameter of the heat kernel.
            
    Returns:
            The heat kernel signature of the graph. The shape of the array is
            (num_vertices).
    """
    eigenvals, eigenvectors = _get_eigenvalues_eigenvectors(adj_mat)
    hks = (np.square(eigenvectors) * np.exp(-diffusion_parameter * eigenvals))\
        .sum(axis=1)
    return hks
    
def _get_eigenvalues_eigenvectors(adj_mat) -> Tuple[Array, Array]:
    """Compute the eigenvector and eigenvalue of the adjacency matrix.
    
    Args:
        adj_mat:
            The adjacency matrix of the graph.
        
    Returns:
            The eigenvectors and eigenvalues of the adjacency matrix.
    """
    # Compute the Laplacian matrix
    laplacian_mat = csgraph.laplacian(adj_mat, normed=True)
    # Compute the eigenvalues and eigenvectors of the Laplacian matrix
    eigenvalues, eigenvectors = eigh(laplacian_mat)
    return eigenvalues, eigenvectors

def graph_extended_persistence_hks(adj_mat: Array,
                                   diffusion_parameter: float = 1.0) \
                                       -> OneHotEncodedPersistenceDiagram:
    """Compute the extended persistence of a graph.
    
    Args:
        adj_mat:
            The adjacency matrix of the graph.
        diffusion_parameter:
            The diffusion parameter of the heat kernel.
            
    Returns:
        Array:
            The extended persistence of the graph.
    """
    # Compute the heat kernel signature
    hks = _heat_kernel_signature(adj_mat, diffusion_parameter)
    # Compute the extended persistence
    persistence_diagram = graph_extended_persistence_gudhi(adj_mat, hks)
    return get_one_hot_encoded_persistence_diagram_from_gudhi_extended(
        persistence_diagram
    )

def graph_extended_persistence_gudhi(A: Array,
                                     filtration_val: Array)\
    -> Tuple[Array, Array, Array, Array]:
    
    # TODO: Rewrite this function
    num_vertices = A.shape[0]
    (xs, ys) = np.where(np.triu(A))
    st = gd.SimplexTree()  # type: ignore
    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)
    for idx, x in enumerate(xs):        
        st.insert([x, ys[idx]], filtration=-1e10)
    for i in range(num_vertices):
        st.assign_filtration([i], filtration_val[i])
    st.make_filtration_non_decreasing()
    st.extend_filtration()
    extended_persistence = st.extended_persistence()
    dgmOrd0, dgmRel1, dgmExt0, dgmExt1 = extended_persistence[0], extended_persistence[1], \
        extended_persistence[2], extended_persistence[3]
    dgmOrd0 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) \
        for p in dgmOrd0 if p[0] == 0]) if len(dgmOrd0) else np.empty([0,2])
    dgmRel1 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) \
        for p in dgmRel1 if p[0] == 1]) if len(dgmRel1) else np.empty([0,2])
    dgmExt0 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) \
        for p in dgmExt0 if p[0] == 0]) if len(dgmExt0) else np.empty([0,2])
    dgmExt1 = np.vstack([np.array([[ min(p[1][0],p[1][1]), max(p[1][0],p[1][1]) ]]) \
        for p in dgmExt1 if p[0] == 1]) if len(dgmExt1) else np.empty([0,2])
    return dgmOrd0, dgmExt0, dgmRel1, dgmExt1
