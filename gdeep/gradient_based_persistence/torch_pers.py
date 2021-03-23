import torch
import networkx as nx
from sklearn.cluster import DBSCAN
from gdeep.gradient_based_persistence.utility import *
from pyvis.network import Network


class smale_quiver:
    def __init__(self,  n_epochs = 10, lr = 0.0001, clusterer = DBSCAN(eps = 0.1,min_samples=5)):
        self.n_epochs = n_epochs
        self.lr = lr
        self.clusterer = clusterer



    def fit(self, class_nn, input_shape, n_sample = 200):
        min_max = get_min_max(class_nn, input_shape, self.n_epochs, n_sample)
        self.quiver = build_graph(min_max, self.clusterer)
        add_index(self.quiver, class_nn(), epsilon= 0.1)

    def fit_transform(self, class_nn, input_shape, n_sample = 200):
        self.fit(class_nn, input_shape, n_sample = 200)
        return self.quiver

    def plot(self):
        nt = Network(notebook=True, directed=True)
        nt.barnes_hut()
        nt.add_nodes([int(i) for i in list(self.quiver.nodes())], value=[float(self.quiver.nodes[i]['loss']) for i in list(self.quiver.nodes())],
                     x=[pos[i][0] for i in list(self.quiver.nodes())], y=[self.quiver.nodes[i]['loss'] for i in list(self.quiver.nodes())],
                     title=['index' + str(self.quiver.nodes[i]['index']) for i in list(self.quiver.nodes())],
                     color=[self.quiver.nodes[i]['index'] for i in list(self.quiver.nodes())])
        for e in list(self.quiver.edges()):
            nt.add_edge(source=int(e[0]), to=int(e[1]))

        nt.show('nx.html')

    def fit_transform_plot(self, class_nn, input_shape, n_sample = 200):
        self.fit(class_nn, input_shape, n_sample = 200)
        self.plot()
        return self.quiver


class persistence:

    def __init__(self, smale_quiver):
        self.smale_quiver = smale_quiver

    def diagram(self):
        one_skeleton = get_one_skeleton(smale_quiver.quiver)
        simplices = get_simplices(one_skeleton)
        return persistence(simplices)

