import numpy
import torch
from gtda.homology import VietorisRipsPersistence, WeakAlphaPersistence
from gtda.graphs import KNeighborsGraph, GraphGeodesicDistance
from typing import List, Any

Array = numpy.ndarray
Tensor = torch.Tensor


def knn_distance_matrix(x: List, k: int = 3):

    """Returns list of distance matrices according
    to the k-NN graph distance of the datasets X

    Args:
        x:
            the point cloud
        k (int):
            the numbers of neighbors for the k-NN graph

    """

    kng = KNeighborsGraph(n_neighbors=k)
    adj = kng.fit_transform(x)
    return GraphGeodesicDistance(directed=False).fit_transform(adj)


def persistence_diagrams_of_activations(
        activations_list: List[Tensor], homology_dimensions=(0, 1), k: int = 5,
        mode: str = "VR", max_edge_length: int = 10) -> List[Any]:
    """Returns list of persistence diagrams of the activations of all
    layers of type layer_types

    Args:
        activations_list (list):
            list of activation
            tensors for each layer
        homology_dimensions (list, optional):
            list of homology
            dimensions. Defaults to `[0, 1]`.
        k (optional) :
            number of neighbors parameter
            of the k-NN distance. If ``k <= 0``, then
            the list of activations is considered
            as a point cloud and no knn distance is
            computed
        mode (optional) :
            choose the filtration ('VR'
            or 'alpha') to compute persistence default to 'VR'.
        max_edge_length (float):
            maximum edge length of the simplices forming
            the complex

    Returns:
        (list):
            list of persistence diagrams of activations
            of the different layers
    """
    for i, activ in enumerate(activations_list):
        if len(activ.shape) > 2:  # in case of non FF layers
            activations_list[i] = activ.view(activ.shape[0], -1)

    if k > 0 and mode == "VR":
        vr = VietorisRipsPersistence(
            homology_dimensions=homology_dimensions,
            metric="precomputed",
            reduced_homology=False,
            infinity_values=20,
            max_edge_length=max_edge_length,
            collapse_edges=True,
        )
        # infinity_values set to avoid troubles with
        # multiple topological features living indefinitely
    elif k <= 0 and mode == "alpha":
        vr = WeakAlphaPersistence(
            homology_dimensions=homology_dimensions,
            reduced_homology=False,
            max_edge_length=max_edge_length,
        )
    else:
        vr = VietorisRipsPersistence(
            homology_dimensions=homology_dimensions,
            reduced_homology=False,
            max_edge_length=max_edge_length,
            collapse_edges=True,
        )

    if k > 0 and mode == "VR":
        activations_list_array = _convert_list_of_tensor_to_numpy(activations_list)
        dist_matrix = knn_distance_matrix(activations_list_array, k=k)
        persistence_diagrams = vr.fit_transform(dist_matrix)
    else:
        activations_list_array = _convert_list_of_tensor_to_numpy(activations_list)
        persistence_diagrams = vr.fit_transform(activations_list_array)

    return persistence_diagrams


def _convert_list_of_tensor_to_numpy(input_list: List[Tensor]) -> List[Array]:
    """private method to convert a list of tensors to a
    list of arrays"""
    output_list: List[Array] = []
    for item in input_list:
        output_list.append(item.detach().cpu().numpy())
    return output_list
