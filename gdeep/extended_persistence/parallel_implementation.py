# %%
from IPython import get_ipython  # type: ignore
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import networkx as nx
import numpy as np

from time import time
from os.path import join

from gdeep.extended_persistence import HeatKernelSignature

#%%

def get_local_minima(adj_mat, filtration_vals):
    """Compute all nodes that have a filtration value that is less than or equal to
    all the filtration values of its neighbors.

    Args:
        adj_mat (np.array): adjacenty matrix of the graph
        filtration_vals (np.array): filtrations values of the nodes

    Returns:
        np.array: boolean array representing the local minima
    """
    min_val_nbs = np.min(filtration_vals * (1.0/adj_mat), axis=1)
    return min_val_nbs >= filtration_vals

def get_directed_graph(adj_mat, filtration_vals):
    """Generated directed graph where the edges are the ones given by the
    adjacency matrix and the direction of the edges are the ones given by
    filtration such that an edge is pointing from a vertex with smaller filtration
    value to a vertex with larger filtration value.

    Args:
        adj_mat (np.array): adjacenty matrix of the graph
        filtration_vals (np.array): filtrations values of the nodes

    Returns:
        nx.DiGraph: directed graph with the direction given by the filtration.
    """
    return nx.from_numpy_matrix(((filtration_vals * (1.0/adj_mat) < filtration_vals.reshape(-1, 1)).T) * 1,
                                create_using=nx.DiGraph)

def get_visited_nodes(adj_mat, filtration_vals):
    """Start a graph traversal starting from all nodes which are local minima in the 
    ascending filtration direction. The traversals passing through the i-th node with the j-th node as
    starting point are indicated by a value of one in the returned array.

    Args:
        adj_mat (np.array): adjacenty matrix of the graph
        filtration_vals (np.array): filtrations values of the nodes

    Returns:
        np.array: array of traversed nodes.
    """
    dgraph = get_directed_graph(adj_mat, filtration_vals)
    graph_size = adj_mat.shape[0]
    visited_nodes = np.zeros((graph_size, graph_size))
    for source in np.argwhere(get_local_minima(adj_mat, filtration_vals)).T[0].tolist():
        visited = list(nx.dfs_preorder_nodes(dgraph, source=source))[1:]
        visited_nodes[source][visited] = 1
    return visited_nodes

def compute_death_times(adj_mat, filtration_vals):
    """Computes the death times for a all connected compontents corresponding to its generator.
    These generators are 
    
    Args:
        adj_mat (np.array): adjacenty matrix of the graph
        filtration_vals (np.array): filtrations values of the nodes

    Returns:
        np.array: array of traversed nodes.
    """
    assert min(filtration_vals) >= 0.0, "Algorithm assumes all filtration values to be positive."
    assert min(filtration_vals) == filtration_vals[0], "Algorithm assumes that the first node has minimal filtration value."
    
    # matrix containing all filtration values of the starting points that went through a
    # given node
    visited_nodes = get_visited_nodes(adj_mat, filtration_vals + 1) * filtration_vals.reshape(-1, 1)

    # minimal filtration value of the starting points that went through a
    # given node
    min_vals = ((visited_nodes == 0) * np.inf + visited_nodes).min(axis=0)

    # 
    death_matrix = visited_nodes * ((visited_nodes != 0) & (visited_nodes > min_vals))

    return (death_matrix != 0).argmax(axis=1)
# %%
# path_to_data = join('tests', 'data')


# # %%
# with open(join(path_to_data, 'xlarge_filtered_graph.npy'), 'rb') as f:
#     adj_mat = np.load(f)
#     filtration_vals = np.load(f)
# del f
# np.count_nonzero(get_local_minima(adj_mat, filtration_vals))

# # %%
# with open(join(path_to_data, 'reddit12k_sample_graph.npy'), 'rb') as f:
#     adj_mat = np.load(f)
# del f
# filtration_vals = HeatKernelSignature(adj_mat, 0.1)()
# assert filtration_vals.shape == np.shape(adj_mat[0],), f"filtration_vals has shape {filtration_vals.shape}"
# #nx.draw(graph, labels={v: f for (v, f) in enumerate(filtration_vals.tolist())},with_labels=True)

# np.count_nonzero(get_local_minima(adj_mat, filtration_vals))

# dgraph = nx.from_numpy_matrix(adj_mat)
# # %%


# # %%
graph_size = 6
adj_mat = np.zeros((graph_size, graph_size))
adj_mat[0, 2] = adj_mat[1, 2] = adj_mat[2, 5] = adj_mat[3, 5] = adj_mat[4, 5] = 1
adj_mat += adj_mat.T
filtration_vals = np.array(list(range(1, graph_size + 1)))
max_filtration = filtration_vals.max()
graph = nx.from_numpy_matrix(adj_mat)
nx.draw(graph, with_labels=True)
assert np.count_nonzero(get_local_minima(adj_mat, filtration_vals)) == graph_size - 2

# %%================================================================
visited_nodes = get_visited_nodes(adj_mat, filtration_vals + 1) #* filtration_vals.reshape(-1, 1)

# minimal filtration value of the starting points that went through a
# given node
visited_nodes[visited_nodes == 0] = np.inf
min_vals = visited_nodes.argmax(axis=0)

# 
visited_nodes[visited_nodes == min_vals] = np.inf


death_times = visited_nodes.argmin(axis=1)
bars = filtration_vals < filtration_vals[death_times]
print(filtration_vals[bars])
print(death_times[bars])

# %%
visited_nodes = get_visited_nodes(adj_mat, filtration_vals + 1) * filtration_vals.reshape(-1, 1)

# minimal filtration value of the starting points that went through a
# given node
min_vals = ((visited_nodes == 0) * np.inf + visited_nodes).min(axis=0)

# 
death_matrix = visited_nodes * ((visited_nodes != 0) & (visited_nodes > min_vals))

# # %%
# from gdeep.extended_persistence.gudhi_implementation import graph_extended_persistence_gudhi
# %timeit graph_extended_persistence_gudhi(adj_mat, filtration_vals)
# %timeit x = compute_death_times(adj_mat, filtration_vals)
# # %%
# %timeit x = compute_death_times(adj_mat, filtration_vals)

# # %%

# %%
