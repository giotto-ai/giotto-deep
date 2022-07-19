import numpy as np
import torch
from gtda.homology import VietorisRipsPersistence

from .. import (OneHotEncodedPersistenceDiagram,
                get_one_hot_encoded_persistence_diagram_from_gtda)


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

    assert pd_filtered.all_close(OneHotEncodedPersistenceDiagram(
        torch.tensor([[0.5000, 0.9000, 0.0000, 1.0000],
            [0.4000, 0.8000, 1.0000, 0.0000]]))
    )

def test_from_numpy():
    x = np.array([
            [0.3, 0.5, 1.0, 0.0],
            [0.4, 0.8, 1.0, 0.0],
            [0.5, 0.9, 0.0, 1.0],
        ])

    x = OneHotEncodedPersistenceDiagram.from_numpy(x)

    assert x.all_close(OneHotEncodedPersistenceDiagram(
                        torch.tensor([[0.3000, 0.5000, 1.0000, 0.0000],
                                      [0.5000, 0.9000, 0.0000, 1.0000],
                                      [0.4000, 0.8000, 1.0000, 0.0000]])
    ))

def test_get_one_hot_encoded_persistence_diagram_from_gtda():
    vr = VietorisRipsPersistence(homology_dimensions=[0, 1])

    # set ranom seed
    np.random.seed(0)

    points = np.random.rand(100, 2)

    diagrams = vr.fit_transform([points])

    diagram = diagrams[0]

    get_one_hot_encoded_persistence_diagram_from_gtda(diagram)._data.shape == (123, 4)
