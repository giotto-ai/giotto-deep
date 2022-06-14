"""Testing for simpleNN."""
# License: GNU AGPLv3

import torch
from torch import nn

from gdeep.models import ModelExtractor
from ..simple_nn import FFNet
from gdeep.data.datasets import DatasetBuilder, DataLoaderBuilder


bd = DatasetBuilder(name="Blobs")
ds_tr, ds_val, _ = bd.build()
dl = DataLoaderBuilder([ds_tr, ds_val])  # type: ignore
dl_tr, dl_val, dl_ts = dl.build()  # sampler=SubsetRandomSampler(train_indices))


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.seqmodel = nn.Sequential(nn.Flatten(), FFNet(arch=[3, 5, 10, 5, 2]))

    def forward(self, x):
        return self.seqmodel(x)


def test_extractor_get_layers_param():
    model = Model1()

    loss_fn = nn.CrossEntropyLoss()
    me = ModelExtractor(model, loss_fn)

    lista = me.get_layers_param()

    for k, item in lista.items():
        k, item.shape  # noqa


def test_extractor_get_decision_boundary():
    model = FFNet(arch=[3, 3])
    loss_fn = nn.CrossEntropyLoss()
    me = ModelExtractor(model, loss_fn)
    x = next(iter(dl_tr))[0][0]
    if x.dtype is not torch.int64:
        res = me.get_decision_boundary(x, n_epochs=5)
        assert res.shape[-1] == 3


def test_extractor_get_activations():
    model = FFNet(arch=[3, 4, 3])
    loss_fn = nn.CrossEntropyLoss()
    me = ModelExtractor(model, loss_fn)
    x = next(iter(dl_tr))[0][0]
    list_activations = me.get_activations(x)
    assert len(list_activations) == 3


def test_extractor_get_gradients():
    model = FFNet(arch=[3, 3, 4, 3])
    loss_fn = nn.CrossEntropyLoss()
    me = ModelExtractor(model, loss_fn)
    batch = next(iter(dl_tr))
    list_activations = me.get_gradients(batch)
    assert len(list_activations) == 2
