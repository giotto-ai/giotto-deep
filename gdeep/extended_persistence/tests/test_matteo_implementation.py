from gdeep.extended_persistence.matteo_implementation import graph_extended_persistence_matteo

from os.path import join
import numpy as np
import pytest

path_to_data = join('gdeep', 'extended_persistence', 'tests', 'data')
    
def test_small_filtered_graph_matteo(benchmark):
    # Load filtered graph
    with open(join(path_to_data, 'small_filtered_graph.npy'), 'rb') as f:
        adj_mat = np.load(f)
        filtration_vals = np.load(f)
    
    result = benchmark(graph_extended_persistence_matteo, *(adj_mat, filtration_vals))
    
    ## Check result
    #assert result ==

def test_medium_filtered_graph_matteo(benchmark):
    # Load filtered graph
    with open(join(path_to_data, 'medium_filtered_graph.npy'), 'rb') as f:
        adj_mat = np.load(f)
        filtration_vals = np.load(f)
    
    result = benchmark(graph_extended_persistence_matteo, *(adj_mat, filtration_vals))
    
    ## Check result
    #assert result ==

def test_large_filtered_graph_matteo(benchmark):
    # Load filtered graph
    with open(join(path_to_data, 'large_filtered_graph.npy'), 'rb') as f:
        adj_mat = np.load(f)
        filtration_vals = np.load(f)
    
    result = benchmark(graph_extended_persistence_matteo, *(adj_mat, filtration_vals))
    
    ## Check result
    #assert result ==

# Implementation is too slow fort the test

# @pytest.mark.benchmark(
#     max_time=10.0,
# )
# def test_xlarge_filtered_graph_matteo(benchmark):
#     # Load filtered graph
#     with open(join(path_to_data, 'xlarge_filtered_graph.npy'), 'rb') as f:
#         adj_mat = np.load(f)
#         filtration_vals = np.load(f)
    
#     result = benchmark(graph_extended_persistence_matteo, *(adj_mat, filtration_vals))
    
    ## Check result
    #assert result ==