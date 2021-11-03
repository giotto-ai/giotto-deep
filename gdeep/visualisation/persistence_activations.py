import torch
from gtda.homology import VietorisRipsPersistence, \
    WeakAlphaPersistence
from gtda.graphs import KNeighborsGraph, \
    GraphGeodesicDistance


def knn_distance_matrix(X, k=3):

    """Returns list of distance matrices according
    to the k-NN graph distance of the datasets X

    Args:
        X (ndarray):
            the point cloud
        k (int):
            the numbers of neighbors for the k-NN graph

    """

    kng = KNeighborsGraph(n_neighbors=k)
    adj = kng.fit_transform(X)
    return GraphGeodesicDistance(directed=False).fit_transform(adj)


def persistence_diagrams_of_activations(activations_list,
                                        homology_dimensions=(0, 1),
                                        k=5,
                                        mode='VR',
                                        max_edge_length=10):
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
            of the k-NN distance .
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
        if len(activ.shape) > 2:  # in caso of non FF layers
            activations_list[i] = activ.view(activ.shape[0], -1)

    if k > 0 and mode == 'VR':
        VR = VietorisRipsPersistence(homology_dimensions=homology_dimensions,
                                     metric='precomputed',
                                     reduced_homology=False,
                                     infinity_values=20,
                                     max_edge_length=max_edge_length,
                                     collapse_edges=True)
        # infinity_values set to avoid troubles with
        # multiple topological features living indefinitely
    elif k == -1 and mode == 'alpha':
        VR = WeakAlphaPersistence(homology_dimensions=homology_dimensions,
                                  reduced_homology=False,
                                  max_edge_length=max_edge_length)
    else:
        VR = VietorisRipsPersistence(homology_dimensions=homology_dimensions,
                                     reduced_homology=False,
                                     max_edge_length=max_edge_length,
                                     collapse_edges=True)

    if k > 0 and mode == 'VR':
        for i, activ in enumerate(activations_list):
            activations_list[i] = activ.cpu()
        dist_matrix = knn_distance_matrix(activations_list,
                                          k=k)
        persistence_diagrams = VR.fit_transform(dist_matrix)
    else:
        persistence_diagrams = VR.fit_transform(activations_list)

    return persistence_diagrams
