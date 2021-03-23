import torch
import torch.nn as nn
import networkx as nx
from sklearn.cluster import DBSCAN
import torch.optim as optim
import numpy as np
from gtda.externals.python.simplex_tree_interface import SimplexTree as ST


def min_max_loss(net, input_shape, n_epochs):
    for param in net.parameters():
        param.requires_grad = False
    inputs_descent = torch.rand(input_shape, requires_grad=True)
    inputs_ascent = inputs_descent.detach().clone()
    inputs_ascent.requires_grad = True

    optimizer = optim.SGD([inputs_descent], lr=0.0001)

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
    optimizer = optim.SGD([inputs_ascent], lr=0.0001)
    for epoch in range(n_epochs):
        outputs = net(inputs_ascent)
        loss = -(outputs).sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        max, loss_max = inputs_ascent, -loss.item()

    return min, loss_min, max, loss_max


def get_min_max(class_nn, input_shape, n_epochs, n_sample):
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
    min_max['min'] = np.nan_to_num(min_max['min'], nan=infinity_max)
    min_max['max'] = np.nan_to_num(min_max['max'], nan=infinity_min)

    return min_max


#### Build Graph

def build_graph(min_max, clusterer):
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


### Hessian

def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)


### Index

def add_index(G, net, epsilon=0.1):
    for i in list(G.nodes()):
        x = torch.tensor(G.nodes[i]['coordinate'], requires_grad=True)
        hessian_x = torch.autograd.functional.hessian(net, x)
        eig = torch.symeig(hessian_x, eigenvectors=False)[0]
        index = sum([1 for i in eig if i < -epsilon])
        G.nodes[i]['index'] = index


def get_one_skeleton(G):
    one_skeleton = nx.Graph()
    for i in list(G.nodes()):
        if G.nodes[i]['index'] == 1:
            if len(list(G.predecessors(i))) == 1:
                one_skeleton.add_node(list(G.predecessors(i))[0])
                one_skeleton.nodes[list(G.predecessors(i))[0]]['value'] = G.nodes[list(G.predecessors(i))[0]]['loss']
            if len(list(G.predecessors(i))) == 2:
                one_skeleton.add_edge(list(G.predecessors(i))[0], list(G.predecessors(i))[1])
                one_skeleton.edges[list(G.predecessors(i))[0], list(G.predecessors(i))[1]]['value'] = G.nodes[i]['loss']
                one_skeleton.nodes[list(G.predecessors(i))[0]]['value'] = G.nodes[list(G.predecessors(i))[0]]['loss']
                one_skeleton.nodes[list(G.predecessors(i))[1]]['value'] = G.nodes[list(G.predecessors(i))[1]]['loss']
    return one_skeleton


def get_simplices(one_skeleton):
    simplices = []
    for e in list(one_skeleton.edges()):
        simplices.append(([e[0], e[1]], one_skeleton.edges[e]['value']))
    for i in list(one_skeleton.nodes()):
        simplices.append(([i], one_skeleton.nodes[i]['value']))
    return simplices


def persistence(simplices):
    filtered_complex = ST()
    for simplex, value in simplices:
        filtered_complex.insert(simplex, value)
    return filtered_complex.persistence()



