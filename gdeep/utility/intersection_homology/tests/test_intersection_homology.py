import numpy as np
import networkx as nx
from ...intersection_homology import IntersectionHomology, \
    compute_coarsest_strata


def test_IntersectionHomology():
    '''check if the class is consistent.
    '''
    ih = IntersectionHomology()
    X = np.random.random((7, 3))
    ih.fit(X)
    ih.transform(X)
    ih.plot()
    ih.transform(X, [0])


def test_IntersectionHomology2():
    '''check if the class is consistent.
    '''
    ih = IntersectionHomology()
    G = nx.Graph()
    edges = [(0, 1), (1, 2), (0, 3), (2, 3), (0, 4),
             (0, 5), (4, 6), (5, 6), (1, 3), (0, 2),
             (0, 6), (4, 5)]
    G.add_edges_from(edges)
    ih.fit(G)
    ih.transform(G)
    ih.plot()
    ih.transform(G, [0])


def test_IntersectionHomology3():
    '''check if the class is consistent.
    '''
    ih = IntersectionHomology(perversity_check=False)
    G = nx.Graph()
    edges = [(0, 1), (1, 2), (0, 3), (2, 3), (0, 4),
             (0, 5), (4, 6), (5, 6), (1, 3), (0, 2),
             (0, 6), (4, 5)]
    G.add_edges_from(edges)
    ih.fit(G)
    ih.transform(G)
    ih.plot()
    ih.transform(G, [0])


def test_use_of_coarse_strata():
    '''check if the option of using the coarsest
    stratification works.
    '''
    ih = IntersectionHomology(coarse_strata=True)
    G = nx.Graph()
    edges = [(0, 1), (1, 2), (0, 3), (2, 3), (0, 4),
             (0, 5), (4, 6), (5, 6), (1, 3), (0, 2),
             (0, 6), (4, 5)]
    G.add_edges_from(edges)
    ih.fit(G)
    ih.transform(G)


def test_IntersectionHomology4():
    '''check if the class is consistent.
    '''
    ih = IntersectionHomology(perversity_check=False)
    ih2 = IntersectionHomology(perversity_check=False)
    G = nx.Graph()
    edges = [(0, 1), (1, 2), (0, 3), (2, 3), (0, 4),
             (0, 5), (4, 6), (5, 6), (1, 3), (0, 2),
             (0, 6), (4, 5)]
    G.add_edges_from(edges)
    ih.fit(G)
    res = ih.transform(G)
    assert res == ih2.fit_transform(G)


def test_low_():
    # unit tests
    M = np.array([[0, 0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 1],
                  [0, 1, 1, 0, 0, 0, 1],
                  [0, 1, 0, 0, 0, 0, 1],
                  [0, 0, 0, 1, 1, 1, 0]])

    assert IntersectionHomology._low_(M, 0) == 0
    assert IntersectionHomology._low_(M, 1) == 5
    assert IntersectionHomology._low_(M, 2) == 4
    assert IntersectionHomology._low_(M, 3) == 6


def test_reduce():
    # unit test
    mat_paper = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 1]
    ])
    mm, _ = IntersectionHomology._reduce(mat_paper)
    assert (mm == np.array([[1, 0, 0, 1],
           [0, 1, 0, 1],
           [0, 0, 1, 1],
           [0, 0, 0, 1],
           [1, 0, 0, 0],
           [1, 0, 1, 0],
           [0, 1, 1, 0],
           [0, 1, 0, 0]])).all()


def test_coarsest_strata():
    # unit test
    simp = [[0], [1], [2], [3], [0, 1],
            [1, 2], [1, 3], [2, 3], [1, 2, 3]]
    assert compute_coarsest_strata(simp) == [[],
                                             [[0, 1]],
                                             [[1, 2, 3]]]
