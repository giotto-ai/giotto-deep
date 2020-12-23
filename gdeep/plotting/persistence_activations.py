import torch
from gtda.homology import VietorisRipsPersistence, WeakAlphaPersistence
from gtda.plotting import plot_diagram, plot_betti_surfaces
from gtda.graphs import KNeighborsGraph, GraphGeodesicDistance
from gdeep.create_nets.utility import get_activations, Layers_list
from gtda.diagrams import BettiCurve
from numpy import inf

def knn_distance_matrix(X,k=3):

    """Returns list of distance matrices according to the k-NN graph distance of the datasets X

    Args:
        X 3D array
        k integer : the numbers of neighbors for the k-NN graph"""


    kng = KNeighborsGraph(n_neighbors=k)
    adj = kng.fit_transform(X)
    return GraphGeodesicDistance(directed=False).fit_transform(adj)

def persistence_diagrams_of_activations(model, X,
                                        layer_types=[torch.nn.Linear],
                                        homology_dimensions=[0, 1], layers=Layers_list('All'), k=-1, mode = 'VR',
                                        max_edge_length = inf):
    """Returns list of persistence diagrams of the activations of all
    layers of type layer_types

    Args:
        model (nn.Module): Neural Network
        X 2darray: [description]
        layer_types (list, optional): [description]. Defaults to [torch.nn.Linear].
        homology_dimensions (list, optional): list of homology dimensions. Defaults to [0, 1].
        layers ([type], optional): list of layer types to consider. Defaults to Layers_list('All')
        k (optional) : number of neighbors parameter of the k-NN distance .
        mode (optional) : choose the filtration ('VR' or 'alpha') to compute persistence default to 'VR'.

    Returns:
        list: list of persistence diagrams of activations of the different layers
    """

    X_tensor = torch.from_numpy(X).float()
    activations_layers = get_activations(model, X_tensor)

    choosen_activations_layers = []
    if k > 0 and mode == 'VR':
        VR = VietorisRipsPersistence(homology_dimensions=homology_dimensions, metric='precomputed',
                                     reduced_homology=False, infinity_values = 20, max_edge_length = max_edge_length,
                                     collapse_edges = True)
        # infinity_values set to avoid troubles with multiple topological features living indefinitely
    elif k == -1 and mode == 'alpha':
        VR = WeakAlphaPersistence(homology_dimensions=homology_dimensions, reduced_homology=False, max_edge_length = max_edge_length)
    else:
        VR = VietorisRipsPersistence(homology_dimensions=homology_dimensions, reduced_homology=False, max_edge_length = max_edge_length,
                                     collapse_edges = True)

    for i, activations_layer in enumerate(activations_layers.get_outputs()):
        if layers.in_list(i):
            choosen_activations_layer = activations_layer.cpu().numpy()
            choosen_activations_layers.append(choosen_activations_layer)

    if k > 0 and mode == 'VR':
        dist_matrix = knn_distance_matrix(choosen_activations_layers, k=k)
        persistence_diagrams = VR.fit_transform(dist_matrix)
    else:
        persistence_diagrams = VR.fit_transform(choosen_activations_layers)

    return persistence_diagrams

def plot_persistence_diagrams(persistence_diagrams, save = False):
    for _, persistence_diagram in enumerate(persistence_diagrams):
        plot_persistence_diagram = plot_diagram(persistence_diagram)

        plot_persistence_diagram.show()

def  betti_plot_layers(persistence_diagrams, homology_dimension = [0,1]):
    """
    Args:
        persistence_diagrams: A list of persistence diagrams of the data accross the layers
        homology_dimension (int array, optional): An array of homology dimensions

    Returns:
          figs/fig – a tuple of figures representing the Betti surfaces of the data accross layers of the NN, with one figure per dimension in homology_dimensions.
           Otherwise, a single figure representing the Betti curve of the single sample present.
    """

    BC = BettiCurve()
    BC.fit(persistence_diagrams)
    plots = plot_betti_surfaces(BC.transform(persistence_diagrams), BC.samplings_)
