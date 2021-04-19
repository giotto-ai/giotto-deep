# %%
import torch
from gdeep.optimisation import PersistenceGradient


# %%
a = torch.tensor(2.0, dtype=torch.float, requires_grad=True)
b = torch.tensor(2.0, dtype=torch.float, requires_grad=True)
c = torch.tensor(3.0, dtype=torch.float, requires_grad=True)


def singleton_tensor(entry, shape=(3, 3)):
    t = torch.zeros(shape)
    t[entry[0], entry[1]] = 1.0
    t.requires_grad = True
    return t


A = (
    a * singleton_tensor((0, 1)) +
    b * singleton_tensor((1, 2)) +
    c * singleton_tensor((0, 2))
    )


A += A.T
hom_dim = (0,)

pg = PersistenceGradient(homology_dimensions=hom_dim, n_epochs=4,
                         lr=0.4, Lambda=0.1, max_edge_length=3,
                         collapse_edges=True, metric="precomputed")

t_loss = pg.persistence_function(A)
t_loss.backward()

var_dict = {"a": a, "b": b, "c": c}
for k, var in var_dict.items():
    print(f"{k} grad: {var.grad}")
# %%
