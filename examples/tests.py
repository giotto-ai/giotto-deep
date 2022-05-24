# %%
import numpy as np
import torch

from gdeep.data.persistence_diagrams import (
    OneHotEncodedPersistenceDiagram,
)

from gdeep.utility.utils import autoreload_if_notebook
autoreload_if_notebook()

# %%

def test_one_hot_encoded_persistence_diagram():
    pd_one_hot = OneHotEncodedPersistenceDiagram(
        torch.tensor([
            [0.3, 0.5, 1.0, 0.0],
            [0.4, 0.8, 1.0, 0.0],
            [0.5, 0.9, 0.0, 1.0],
        ])
    )

    lifetimes = pd_one_hot.get_lifetimes()

    assert torch.allclose(lifetimes, torch.tensor([0.2000, 0.4000, 0.4000]))

    pd_filtered = pd_one_hot.filter_by_lifetime(min_lifetime=0.3, max_lifetime=0.5)

    assert torch.allclose(pd_filtered,
        torch.tensor([[0.5000, 0.9000, 0.0000, 1.0000],
            [0.4000, 0.8000, 1.0000, 0.0000]]))

# %%
x = np.array([
            [0.3, 0.5, 1.0, 0.0],
            [0.4, 0.8, 1.0, 0.0],
            [0.5, 0.9, 0.0, 1.0],
        ])

OneHotEncodedPersistenceDiagram.from_numpy(x)

assert torch.allclose(OneHotEncodedPersistenceDiagram.from_numpy(x),
                      torch.tensor([[0.3000, 0.5000, 1.0000, 0.0000],
                                    [0.5000, 0.9000, 0.0000, 1.0000],
                                    [0.4000, 0.8000, 1.0000, 0.0000]])
)
# %%
y = torch.tensor([
            [0.3, 0.5, 1.0, 0.0],
            [0.4, 0.8, 1.0, 0.0],
            [0.5, 0.9, 0.0, 1.0],
        ])
OneHotEncodedPersistenceDiagram(y)
# %%
from gtda.plotting.persistence_diagrams import plot_diagram

pd =  torch.tensor\
      ([[0.0928, 0.0995, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0916, 0.1025, 1.0000, 0.0000, 0.0000, 0.0000],
        [0.0978, 0.1147, 1.0000, 0.0000, 0.0000, 0.0000],
        [0.0978, 0.1147, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0916, 0.1162, 0.0000, 0.0000, 0.0000, 1.0000],
        [0.0740, 0.0995, 1.0000, 0.0000, 0.0000, 0.0000],
        [0.0728, 0.0995, 1.0000, 0.0000, 0.0000, 0.0000],
        [0.0740, 0.1162, 0.0000, 0.0000, 0.0000, 1.0000],
        [0.0728, 0.1162, 0.0000, 0.0000, 1.0000, 0.0000],
        [0.0719, 0.1343, 0.0000, 0.0000, 0.0000, 1.0000],
        [0.0830, 0.2194, 1.0000, 0.0000, 0.0000, 0.0000],
        [0.0830, 0.2194, 1.0000, 0.0000, 0.0000, 0.0000],
        [0.0719, 0.2194, 0.0000, 1.0000, 0.0000, 0.0000]])
      
pd = OneHotEncodedPersistenceDiagram(pd)

pd = pd.detach().numpy()

# %%

# convert one-hot encoding to categorical encoding
pd_categorical = pd[:, 2:].argmax(axis=1)

# %%
fig = plot_diagram(np.concatenate([pd[:, :2], pd_categorical.reshape(-1, 1)], axis=1))
# %%
fig.t