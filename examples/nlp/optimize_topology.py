# %%
import numpy as np
import torch
from gdeep.optimisation import PersistenceGradient


# %%
a = torch.tensor(2.0, dtype=torch.float, requires_grad=True)
b = torch.tensor(2.0, dtype=torch.float, requires_grad=True)
c = torch.tensor(3.0, dtype=torch.float, requires_grad=True)


def singleton_tensor(entry, shape=(3, 3)):
    t = torch.zeros(shape)
    t[entry[0], entry[1]] = 1.0
    return t


A = (
    a * singleton_tensor((0, 1)) +
    b * singleton_tensor((1, 2)) +
    c * singleton_tensor((0, 2))
    )


A += A.T

print("Adjacency matrix:\n", A)

hom_dim = (0,1)

pg = PersistenceGradient(homology_dimensions=hom_dim, n_epochs=4,
                         lr=0.4, Lambda=0., max_edge_length=3,
                         collapse_edges=True, metric="precomputed")

t_loss = pg.persistence_function(A)
t_loss.backward()

var_dict = {"a": (a, -1.0), "b": (b, -1.0), "c": (c, 0.0)}
for k, var in var_dict.items():
    print(f"{k} grad: {var[0].grad}")
    np.testing.assert_almost_equal(var[0].grad.item(), var[1])
# %%

a = torch.tensor(1.0, dtype=torch.float, requires_grad=True)
b = torch.tensor(1.0, dtype=torch.float, requires_grad=True)
c = torch.tensor(2.0, dtype=torch.float, requires_grad=True)
d = torch.tensor(2.0, dtype=torch.float, requires_grad=True)


def singleton_tensor(entry, shape=(4, 4)):
    t = torch.zeros(shape)
    t[entry[0], entry[1]] = 1.0
    return t


A = (
    a  * singleton_tensor((1, 3)) +
    b  * singleton_tensor((2, 3)) +
    c  * singleton_tensor((1, 2)) +
    d  * singleton_tensor((0, 1)) +
    10 * singleton_tensor((0, 2)) +
    10 * singleton_tensor((0, 3))
    )


A += A.T

print("Adjacency matrix:\n", A)

hom_dim = (0,)

pg = PersistenceGradient(homology_dimensions=hom_dim, n_epochs=4,
                         lr=0.4, Lambda=0., max_edge_length=3,
                         collapse_edges=True, metric="precomputed")

t_loss = pg.persistence_function(A)
t_loss.backward()

var_dict = {"a": (a, -1.0), "b": (b, -1.0), "c": (c, 0.0), "d": (d, -1.0)}
for k, var in var_dict.items():
    print(f"{k} grad: {var[0].grad}")
    np.testing.assert_almost_equal(var[0].grad.item(), var[1])
# %%
[1, 2].index(3)