# %%
import torch

from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import (
    OneHotEncodedPersistenceDiagram,
)
from gdeep.utility import autoreload_if_notebook

autoreload_if_notebook()

# %%

pd_one_hot = OneHotEncodedPersistenceDiagram(
    torch.tensor([
        [0.3, 0.5, 1.0, 0.0],
        [0.4, 0.8, 1.0, 0.0],
        [0.5, 0.9, 0.0, 1.0],
    ])
)

# %%
lifetimes = pd_one_hot.get_lifetimes()

pd_one_hot.filter_by_lifetime(min_lifetime=0.3, max_lifetime=0.5)
# %%

lifetimes_copy = torch.Tensor(lifetimes)
type(lifetimes_copy)
# %%
x = torch.tensor([1.0, 2.0, 1.0, 0.5])
x - pd_one_hot
# %%
type(pd)
# %%
torch.is_tensor(x)
# %%
