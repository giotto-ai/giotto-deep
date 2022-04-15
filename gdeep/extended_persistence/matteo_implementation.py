import numpy as np
from copy import deepcopy
import networkx as nx
from networkx import algorithms as algo

class GraphExtendedPersistenceMatteo():
    def __init__(self, adj_mat, filtration_vals):
        # create graph from matrix
        g_full = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph)

        # filter the graph with f
        self.graph = nx.DiGraph()

        # filtration
        self.filtration_vals = np.argsort(filtration_vals)
    
    def get_Ord0(self):
        return None
    
    def get_Ext0(self):
        return None
    
    def get_Rel1(self):
        return None
    
    def get_Ext1(self):
        return None
    
    def get_all_diagrams(self):
        return self.get_Ord0(), self.get_Ext0(), self.get_Rel1(), self.get_Ext1()
        
        
def graph_extended_persistence_matteo(A, f):
    """This functions computes the extended persistence
    of graphs.
    
    Args:
        A (np.array):
            adjacency matrix
        f (np.array):
            array of dimension one,
            in which the value at position
            ``i``corresponds to the f-value at
            node ``i``.
            
    Retuns:
        np.array:
            The array is a persistent diagrams in
            the format of giotto-tda."""
    # init
    PD0 = []
    PD1 = []
    previous_roots = []
    b = dict()
    d = dict()

    # create graph from matrix
    g_full = nx.from_numpy_matrix(A, create_using=nx.DiGraph)

    # filter the graph with f
    g = nx.DiGraph()

    # filtration
    filt = np.argsort(f)

    # PD0, part 1
    # the graph being built with ascending values
    for node in filt:
        g.add_node(node)
        for neigh in g_full.neighbors(node):
            if g.has_node(neigh):
                if f[node] >= f[neigh]:
                    g.add_edge(neigh, node)
        #print("Edges: ", g.edges)
        # now the graph is at time t
        all_roots = {n for n,d in g.in_degree() if d == 0}
        G = g.to_undirected()
        cc = nx.connected_components(G)
        roots = []
        for component in cc:
            intersection = list(all_roots.intersection(component))
            if intersection:
                roots.append(intersection[0])
        #print("Roots: ", roots)
        new_borns = list(set(roots).difference(set(previous_roots)))
        for r in new_borns:
            b[r] = f[node]
        deads = list(set(previous_roots).difference(set(roots)))
        for r in deads:
            d[r] = f[node]
        del(previous_roots)
        previous_roots = deepcopy(roots)
        del(roots)

    for dd in d.keys():   
        for bb in b.keys():
            if dd == bb:
                PD0.append((b[bb], d[dd]))

    # PD0, part 2
    for component in nx.connected_components(G):
        death = max(f[np.array(list(component))])
        birth = min(f[np.array(list(component))])
        PD0.append((birth, death))

    # PD1, part 1
    for cycle in nx.cycle_basis(G):
        PD1.append((max(f[cycle]), min(f[cycle])))

    
    # the graph of the relative homology is being built top-down
    in_path = lambda e, path: (e[0], e[1]) in path or (e[1], e[0]) in path
    cycle_to_path = lambda path: list(zip(path+path[:1], path[1:] + path[:1]))
    in_a_cycle = lambda e, cycle: in_path(e, cycle_to_path(cycle))

    # PD1, part 2
    old_node = None
    for node in np.flip(filt):
        if old_node is not None:
            if (old_node, node) not in G.edges or (node, old_node) not in G.edges:
                G.add_edge(old_node, node)
                cycles = [cycle for cycle in nx.cycle_basis(G) if in_a_cycle((old_node, node),cycle)]
                #print("Cycles:", cycles)
                for cycle in cycles:
                    PD1.append((f[node], min(f[cycle])))
        old_node = node



    #print("PD0:", PD0)
    #print("PD1:", PD1)

    output = []

    for pair in PD0:
        output.append((*pair, 0))
    for pair in PD1:
        output.append((*pair, 1))

    return np.array([output])