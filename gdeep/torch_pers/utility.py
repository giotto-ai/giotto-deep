import torch
import torch.nn as nn
import networkx as nx
from sklearn.cluster import DBSCAN
import torch.optim as optim
import numpy as np
from gtda.externals.python.simplex_tree_interface import SimplexTree as ST


def min_max_loss(net, input_shape, n_epochs, lr=0.0001):
    '''

    Args:
        net: a nn.module
        input_shape: the input_shape for net
        n_epochs: the number of epochs for which to run gradient ascent/descent
        lr: learning rate of gradient ascent/descent

    Returns: from a random initialization, returns the endpoints of gradient descent and ascent, together
    with the respective values of the function

    '''


    for param in net.parameters():
        param.requires_grad = False
    inputs_descent = torch.rand(input_shape, requires_grad=True)
    inputs_ascent = inputs_descent.detach().clone()
    inputs_ascent.requires_grad = True

    optimizer = optim.SGD([inputs_descent], lr=lr)

    # perform gradient descent
    for epoch in range(n_epochs):
        # forward + backward + optimize
        outputs = net(inputs_descent)
        optimizer.zero_grad()
        loss = (outputs).sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        min, loss_min = inputs_descent, loss.item()

    # perform gradient ascent
    optimizer = optim.SGD([inputs_ascent], lr=lr)
    for epoch in range(n_epochs):
        outputs = net(inputs_ascent)
        loss = - outputs.sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        max, loss_max = inputs_ascent, -loss.item()

    return min, loss_min, max, loss_max


def get_min_max(class_nn, input_shape, n_epochs, n_sample):

    '''

    Args:
        class_nn: a class of nn.module encoding the function to study
        input_shape: the input_shape of class_nn
        n_epochs: the number of epochs for which to run gradient ascent/descent
        n_sample: the number of sample gradient flow to perform

    Returns: a list of starting point and endpoint of integral lines with respect to the gradient
    of class_nn, together with their respective function values

    '''
    min_max = {'min': [], 'loss_min': [], 'max': [], 'loss_max': []}
    net = class_nn()
    # net.to(dev)
    for i in range(n_sample):
        min, loss_min, max, loss_max = min_max_loss(net, input_shape, n_epochs)
        min_max['min'].append(min)
        min_max['loss_min'].append(loss_min)
        min_max['max'].append(max)
        min_max['loss_max'].append(loss_max)
    infinity_max = [1000000.0] * len(min_max['min'][0])
    infinity_min = [-1000000.0] * len(min_max['max'][0])
    min_max['min'] = torch.stack(min_max['min']).detach()
    min_max['max'] = torch.stack(min_max['max']).detach()
    min_max['min'] = np.nan_to_num(min_max['min'], nan=infinity_min)
    min_max['max'] = np.nan_to_num(min_max['max'], nan=infinity_max)

    return min_max


#### Build Graph

def build_graph(min_max, clusterer = DBSCAN(eps = 0.1,min_samples=5)):
    '''

    Args:
        min_max: an output of get_min_max function
        clusterer: a clusterer class in the sci-kit learn api

    Returns: A networkx grqph, with nodes being clustered critical points, and an edges between two critical
    points if we found an integral line between those points

    '''

    G = nx.DiGraph()

    #### Cluster nodes of the graph
    nodes = min_max['min']
    n_mins = len(min_max['min'])
    nodes = np.append(min_max['min'], min_max['max'], axis=0)
    labels = clusterer.fit_predict(nodes)
    edges = [(labels[i], labels[i + n_mins]) for i in range(len(min_max['min'])) if
             labels[i] != -1 and labels[i + n_mins] != -1]
    G.add_edges_from(edges)
    all_loss = np.append(min_max['loss_min'], min_max['loss_max'])
    for i in list(G.nodes()):
        if (i, i) in list(G.edges()):
            G.remove_edge(i, i)
        loss_mean = np.nanmean(all_loss[labels == i])
        coordinate_mean = np.nanmean(nodes[labels == i], axis=0)
        if not np.isnan(loss_mean):
            G.nodes[i]['loss'] = loss_mean
            G.nodes[i]['coordinate'] = coordinate_mean
        else:
            G.remove_node(i)

    return G



### Index

def add_index(G, net, threshold=0.1):
    '''

    Args:
        G: an output of build_graph
        net: the function from which G is built
        threshold: a smoothing parameter to decide when an eigenvalue of the hessian must be considered negative

    Returns:

    '''
    for i in list(G.nodes()):
        x = torch.tensor(G.nodes[i]['coordinate'], requires_grad=True)
        hessian_x = torch.autograd.functional.hessian(net, x)
        eig = torch.symeig(hessian_x, eigenvectors=False)[0]
        index = sum([1 for value in eig if value < -threshold])
        G.nodes[i]['index'] = index





