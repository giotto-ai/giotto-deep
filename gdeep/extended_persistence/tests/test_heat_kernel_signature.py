from gdeep.extended_persistence.heat_kernel_signature import \
    _heat_kernel_signature, _get_eigenvalues_eigenvectors, \
    graph_extended_persistence_hks

import numpy as np


def test_sample_mutag_graph():
    adj_mat = np.array(
       [
       [0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0. , 0.],
       [1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       ],
       dtype=np.float32)
    
    eigenvalues, eigenvectors = _get_eigenvalues_eigenvectors(adj_mat)
    
    for time in [0.1, 1.0, 10]:
        hks: np.ndarray = _heat_kernel_signature(adj_mat, time)
        assert hks.shape == (adj_mat.shape[0],)
        assert np.allclose(hks,
                        np.square(eigenvectors)\
                            .dot(np.diag(np.exp(-time * eigenvalues)))\
                                .sum(axis=1)), \
                                    "Heat kernel signature is not correct"
                                    
def test_extended_persistence_hks_empty_diagram():
    adj_mat = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert graph_extended_persistence_hks(adj_mat).shape == (0, 6)
    
def test_extended_persistence_hks_small_diagram():
    adj_mat = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    extended_persistence_hks = graph_extended_persistence_hks(adj_mat)
    assert extended_persistence_hks.shape == (2, 6)
    
    expected_diagram_hks = np.array([
    [0.46777354, 0.56766764, 1.        , 0.        , 0.        , 0.        ],
    [0.46777354, 0.56766764, 0.        , 1.        , 0.        , 0.        ]]
    )
    
    assert np.allclose(extended_persistence_hks, expected_diagram_hks)
    
def test_extended_persistence_mutag_sample_graph():
    perslay_persistence_diagrams = [(0.0722339, 0.22594431),
                                    (0.08453899, 0.22594431),
                                    (0.08453902, 0.22594431),
                                    (0.10453895, 0.19677208),
                                    (0.10453895, 0.19677208),
                                    (0.10453902, 0.15688537),
                                    (0.10453902, 0.19677208)]
    
    adj_mat = np.array(
      [[0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
       [1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 1., 0., 0., 0., 0., 0., 1.],
       [0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 1., 1., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])
    
    computed_diagram = []
    for birth, death in graph_extended_persistence_hks(adj_mat, 10)[:, :2]:
        if birth > 1e-5 and death > 1e-5:
            computed_diagram.append((birth, death))

    # sort points_new by first element
    computed_diagram.sort(key=lambda x: x[0])
    
    assert len(perslay_persistence_diagrams) == len(computed_diagram)
    
    for point_perslay, point_computed in \
        zip(computed_diagram, perslay_persistence_diagrams):
        assert np.allclose(point_perslay, point_computed, atol=1e-5)