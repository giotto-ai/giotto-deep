import numpy as np
import plotly.express as px
import torch
from gdeep.optimisation import PersistenceGradient


def test_PersistenceGradient_2d():
    '''check if the class is consistent.
    '''
    X = torch.tensor([[1,0.],[0,1.],[2,2],[2,1]])
    hom_dim = (0,1)
    pg = PersistenceGradient(homology_dimensions=hom_dim,
                         n_epochs=4,
                         lr=0.4,Lambda=0.1,
                         max_edge_length=3,
                         collapse_edges=True)
    assert X.dtype == torch.float32
    assert pg.Phi(X).shape[0] == 14
    assert (np.array([[ 0,  4],[ 1,  5],
            [ 2,  6],[ 3, -1],[-1, -1],
            [-1, -1],[-1, -1]]) == pg._persistence(X)[0]).all()
    assert pg.persistence_function(X).item() >= -2.328427314758301 -0.001
    pg.SGD(X)
    
def test_PersistenceGradient_3d():
    '''check if the class is consistent.
    '''
    X = torch.tensor([[1,0.,1],[0,1.,0],[2,2,1],[2,1,2]])
    hom_dim = (0,1)
    pg = PersistenceGradient(homology_dimensions=hom_dim,
                         n_epochs=4,
                         lr=0.4,Lambda=0.1,
                         max_edge_length=3,
                         collapse_edges=True)
    assert X.dtype == torch.float32
    assert pg.Phi(X).shape[0] == 14
    assert (np.array([[ 0,  4],[ 1,  5],
            [ 2,  6],[ 3, -1],[-1, -1],
            [-1, -1],[-1, -1]]) == pg._persistence(X)[0]).all()
    assert pg.persistence_function(X).item() >= -2.7783148288726807 -0.001
    pg.SGD(X)
    
def test_PersistenceGradient_5d():
    '''check if the class is consistent.
    '''
    X = torch.tensor([[1,0.,1,0.5,1],[0,1.,0,0.5,1],[2,2,1,0.5,1],[2,1,2,0.5,1]])
    hom_dim = (0,1)
    pg = PersistenceGradient(homology_dimensions=hom_dim,
                         n_epochs=4,
                         lr=0.4,Lambda=0.1,
                         max_edge_length=3,
                         collapse_edges=True)
    assert X.dtype == torch.float32
    assert pg.Phi(X).shape[0] == 14
    assert (np.array([[ 0,  4],[ 1,  5],
            [ 2,  6],[ 3, -1],[-1, -1],
            [-1, -1],[-1, -1]]) == pg._persistence(X)[0]).all()
    assert pg.persistence_function(X).item() >= -2.7783148288726807 -0.001
    pg.SGD(X)
    
def test_PersistenceGradient_4d():
    '''check if the matrix input works properly'''
    X = torch.tensor([[1,0.,1,0.5],[0,1.,0,0.5],[2,2,1,0.5],[2,1,2,0.5]])
    hom_dim = (0,1)
    pg = PersistenceGradient(homology_dimensions=hom_dim,
                         n_epochs=4,
                         lr=0.4,Lambda=0.1,
                         max_edge_length=3,
                         metric="precomputed",
                         collapse_edges=False)
    assert X.dtype == torch.float32
    assert pg.Phi(X).shape[0] == 14
    print(pg._persistence(X)[0])
    assert (np.array([[ 0,  1],[-1, -1]]) == pg._persistence(X)[0]).all()
    assert pg.persistence_function(X).item() >= 0.3467579483985901 + 0.001

    pg.SGD(X)
