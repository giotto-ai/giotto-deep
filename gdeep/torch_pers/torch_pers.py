import torch
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from gdeep.torch_pers.utility import *
from pyvis.network import Network


class SmaleQuiverComputer:
    '''
    This class computes the Smale Quiver associated to a function encoded
    within a nn.module, as a networkx graph

    Args:  - n_epochs (int) : the number of epochs for which to run gradient ascent/descent
    - lr (float) : the learning rate of the gradient ascent/descent steps
    - clusterer : a clustering class, in the sci-kit learn api

    '''


    def __init__(self,  n_epochs = 10, lr = 0.0001, clusterer = DBSCAN(eps = 0.1,min_samples=5)):
        self.n_epochs = n_epochs
        self.lr = lr
        self.clusterer = clusterer



    def fit(self, class_nn, input_shape, n_sample = int(200)):

        ''' Computes the smale quiver of the fonction encoded by class_nn
        Stores it in the attribute .quiver

        Args:
            class_nn (nn.module) : a class of nn.module encoding the function to study
            input_shape (array) : array describing the input shape of the function
            n_sample (optional, default = 200) : the number of points to sample

        Returns:

        '''


        min_max = get_min_max(class_nn, input_shape, self.n_epochs, n_sample)
        self.quiver = build_graph(min_max, self.clusterer)
        add_index(self.quiver, class_nn(), epsilon= 0.1)
        nt = Network(notebook=True, directed=True)
        nt.barnes_hut()
        nt.add_nodes([int(i) for i in list(self.quiver.nodes())],
                     value=[float(self.quiver.nodes[i]['loss']) for i in list(self.quiver.nodes())],
                     title=['index' + str(self.quiver.nodes[i]['index']) for i in list(self.quiver.nodes())],
                     color=[self.quiver.nodes[i]['index'] for i in list(self.quiver.nodes())])
        for e in list(self.quiver.edges()):
            nt.add_edge(source=int(e[0]), to=int(e[1]))
        self.quiver_to_plot = nt

    def fit_transform(self, class_nn, input_shape, n_sample = int(200)):
        ''' Computes the smale quiver of the fonction encoded by class_nn
        Stores it in the attribute .quiver and returns it

        Args:
            class_nn (nn.module) : a class of nn.module encoding the function to study
            input_shape (array) : array describing the input shape of the function
            n_sample (optional, default = 200) : the number of points to sample

        Returns:

        '''
        self.fit(class_nn, input_shape, n_sample = n_sample)
        return self.quiver


    def fit_transform_plot(self, class_nn, input_shape, n_sample = int(200)):
        self.fit(class_nn, input_shape, n_sample = n_sample)
        self.plot()
        return self.quiver


class PersistenceComputer:

    '''
        Given a smale quiver, this class computes the associated 0th barcode
    '''

    def __init__(self, smale_quiver):
        self.smale_quiver = smale_quiver

    def get_one_skeleton(self):
        '''

        Returns: The filtered graph, where edges are one critical points of the function,
        connecting the 0-critical points

        '''

        one_skeleton = nx.Graph()
        for i in list(self.smale_quiver.nodes()):
            if self.smale_quiver.nodes[i]['index'] == 1:
                if len(list(self.smale_quiver.predecessors(i))) == 1:
                    one_skeleton.add_node(list(self.smale_quiver.predecessors(i))[0])
                    one_skeleton.nodes[list(self.smale_quiver.predecessors(i))[0]]['value'] = self.smale_quiver.nodes[list(self.smale_quiver.predecessors(i))[0]][
                        'loss']
                if len(list(self.smale_quiver.predecessors(i))) == 2:
                    one_skeleton.add_edge(list(self.smale_quiver.predecessors(i))[0], list(self.smale_quiver.predecessors(i))[1])
                    one_skeleton.edges[list(self.smale_quiver.predecessors(i))[0], list(self.smale_quiver.predecessors(i))[1]]['value'] = self.smale_quiver.nodes[i][
                        'loss']
                    one_skeleton.nodes[list(self.smale_quiver.predecessors(i))[0]]['value'] = self.smale_quiver.nodes[list(self.smale_quiver.predecessors(i))[0]][
                        'loss']
                    one_skeleton.nodes[list(self.smale_quiver.predecessors(i))[1]]['value'] = self.smale_quiver.nodes[list(self.smale_quiver.predecessors(i))[1]][
                        'loss']
        self.one_skeleton = one_skeleton

    def get_simplices(self):
        '''

        Returns: Utility function to build the filtration for SimplexTree

        '''
        self.get_one_skeleton()
        simplices = []
        for e in list(self.one_skeleton.edges()):
            simplices.append(([e[0], e[1]], self.one_skeleton.edges[e]['value']))
        for i in list(self.one_skeleton.nodes()):
            simplices.append(([i], self.one_skeleton.nodes[i]['value']))
        self.simplices = simplices


    def get_persistence(self):
        '''

        Returns: 0th Bardcode of the smale quiver

        '''
        self.get_simplices()
        filtered_complex = ST()
        for simplex, value in self.simplices:
            filtered_complex.insert(simplex, value)
        pers = filtered_complex.persistence()
        d = []
        for dim , bar in pers:
            d.append([bar[0],bar[1],dim])
        self.persistence = np.array(d)
        return self.persistence



