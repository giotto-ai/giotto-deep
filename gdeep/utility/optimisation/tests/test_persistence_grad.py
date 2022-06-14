
import torch
from ...optimisation import PersistenceGradient


def test_persistence_gradient_2d():
    """check if the class is consistent.
    """
    X = torch.tensor([[1, 0.], [0, 1.], [2, 2], [2, 1]])
    hom_dim = [0, 1]
    pg = PersistenceGradient(homology_dimensions=hom_dim,
                             zeta=0.1,
                             max_edge_length=3,
                             collapse_edges=False)

    assert X.dtype == torch.float32
    assert pg.phi(X).shape[0] == 14

    assert pg.persistence_function(X).item() >= -2.328427314758301 - 0.001
    pg.sgd(X, n_epochs=4, lr=0.4)


def test_persistence_gradient_3d():
    """check if the class is consistent.
    """
    X = torch.tensor([[1, 0., 1], [0, 1., 0], [2, 2, 1], [2, 1, 2]])
    hom_dim = [0, 1]
    pg = PersistenceGradient(homology_dimensions=hom_dim,
                             zeta=0.1,
                             max_edge_length=3,
                             collapse_edges=False)
    assert X.dtype == torch.float32
    assert pg.phi(X).shape[0] == 14

    assert pg.persistence_function(X).item() >= -2.7783148288726807 - 0.001
    pg.sgd(X, n_epochs=4, lr=0.4)


def test_persistence_gradient_5d():
    """check if the class is consistent.
    """
    X = torch.tensor([[1, 0., 1, 0.5, 1], [0, 1., 0, 0.5, 1],
                      [2, 2, 1, 0.5, 1], [2, 1, 2, 0.5, 1]])
    hom_dim = [0, 1]
    pg = PersistenceGradient(homology_dimensions=hom_dim,
                             zeta=0.1,
                             max_edge_length=3,
                             collapse_edges=False)
    assert X.dtype == torch.float32
    assert pg.phi(X).shape[0] == 14
    assert pg.persistence_function(X).item() >= -2.7783148288726807 - 0.001
    pg.sgd(X, n_epochs=4, lr=0.4)


def test_persistence_gradient_4d():
    """check if the matrix input works properly"""
    X = torch.tensor([[1, 0., 1, 0.5], [0, 1., 0, 0.5],
                      [2, 2, 1, 0.5], [2, 1, 2, 0.5]])
    hom_dim = [0, 1]
    pg = PersistenceGradient(homology_dimensions=hom_dim,
                             zeta=0.1,
                             max_edge_length=3,
                             metric="precomputed",
                             collapse_edges=False)
    assert X.dtype == torch.float32
    assert pg.phi(X).shape[0] == 14
    assert pg.persistence_function(X).item() >= 0.3467579483985901 + 0.001
    pg.sgd(X)


def test_persistence_gradient_matrix():
    # simulate the weighted graph
    dist = torch.tensor([[0., 2, 3],
                        [2, 0., 2.2],
                        [3, 2.2, 0.]])

    pg = PersistenceGradient(homology_dimensions=[0, 1],
                             zeta=0.0,
                             collapse_edges=False,
                             metric="precomputed")
    assert all(pg.phi(dist) == torch.tensor([0., 0., 0., 2., 2.2, 3., 3.]))
    assert pg.persistence_function(dist).item() >= -6.3 - 0.0001
    pg.sgd(dist, n_epochs=1, lr=1)


def test_persistence_gradient_matrix_2():
    # simulate the weighted graph
    dist = torch.tensor([[0., 2., 10., 10.],
                         [2., 0., 2., 1],
                         [10., 2., 0., 1],
                         [10., 1, 1, 0.]])
    pg = PersistenceGradient(homology_dimensions=[0, 1],
                             zeta=0.0,
                             collapse_edges=False,
                             metric="precomputed")
    assert all(pg.phi(dist) == torch.tensor([0., 0., 0., 0., 1.,
                                             1., 2., 2., 2., 10.,
                                             10., 10., 10., 10.]))
    assert pg.persistence_function(dist).item() > -23.
    pg.sgd(dist, n_epochs=1, lr=0.002)
                                  

def test_persistence_gradient_pts():
    # test explicit gradients
    pts = torch.tensor([[0., 0.],
                        [0., 1.],
                        [1., 0.]])
    pg = PersistenceGradient(homology_dimensions=[0, 1],
                             zeta=0.0,
                             collapse_edges=False,
                             metric="euclidean")

    assert all(pg.phi(pts)[:5] == torch.tensor([0., 0., 0.,
                                                1., 1.]))

    pg.sgd(pts, n_epochs=1, lr=0.002)

    assert (pts.grad == torch.tensor([[1.,  1.],
                                      [0., -1.],
                                      [-1.,  0.]])).all().item()
