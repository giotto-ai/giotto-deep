from gdeep.extended_persistence.gudhi_implementation import graph_extended_persistence_gudhi

from os.path import join
import numpy as np
import time

path_to_data = join('gdeep', 'extended_persistence', 'tests', 'data')
    
def test_small_filtered_graph_gudhi(benchmark):
    # Load filtered graph
    with open(join(path_to_data, 'small_filtered_graph.npy'), 'rb') as f:
        adj_mat = np.load(f)
        filtration_vals = np.load(f)
    
    result = benchmark(graph_extended_persistence_gudhi, *(adj_mat, filtration_vals))
    
    ## Check result
    #assert result ==

def test_medium_filtered_graph_gudhi(benchmark):
    # Load filtered graph
    with open(join(path_to_data, 'medium_filtered_graph.npy'), 'rb') as f:
        adj_mat = np.load(f)
        filtration_vals = np.load(f)
    
    result = benchmark(graph_extended_persistence_gudhi, *(adj_mat, filtration_vals))
    
    ## Check result
    #assert result ==

def test_large_filtered_graph_gudhi(benchmark):
    # Load filtered graph
    with open(join(path_to_data, 'large_filtered_graph.npy'), 'rb') as f:
        adj_mat = np.load(f)
        filtration_vals = np.load(f)
    
    assert adj_mat.shape[0] == 200
    
    result = benchmark(graph_extended_persistence_gudhi, *(adj_mat, filtration_vals))
    
    ## Check result
    #assert result ==

def test_xlarge_filtered_graph_gudhi(benchmark):
    # Load filtered graph
    with open(join(path_to_data, 'xlarge_filtered_graph.npy'), 'rb') as f:
        adj_mat = np.load(f)
        filtration_vals = np.load(f)
    assert adj_mat.shape[0] == 1_000
    
    result = benchmark(graph_extended_persistence_gudhi, *(adj_mat, filtration_vals))
    
    ## Check result
    #assert result ==