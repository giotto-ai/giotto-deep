import torch

from gdeep.trainer import accuracy


def test_accuracy():
    predictions = torch.rand((10, 3))
    y = torch.randint(0, 2, (10,))

    acc = accuracy(predictions, y)

    assert acc >= 0
