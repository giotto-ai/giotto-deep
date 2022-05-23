"""Testing for simpleNN."""
# License: GNU AGPLv3
import pytest
import torch
from torch import nn

from gdeep.models import ModelExtractor
from ..simple_nn import FFNet
from gdeep.data.datasets import DatasetBuilder, DataLoaderBuilder

bd = DatasetBuilder(name="Blobs")
ds_tr, ds_val, _ = bd.build()
#train_indices = list(range(160))
dl = DataLoaderBuilder((ds_tr, ds_val))
dl_tr, dl_val, dl_ts = dl.build() #, sampler=SubsetRandomSampler(train_indices))

class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.seqmodel = nn.Sequential(nn.Flatten(), FFNet(arch=[3, 5, 10, 5, 2]))
    def forward(self, x):
        return self.seqmodel(x)


def test_extractor1():
    model = model1()

    loss_fn = nn.CrossEntropyLoss()
    me = ModelExtractor(model, loss_fn)

    lista = me.get_layers_param()

    for k, item in lista.items():
        k, item.shape

def test_extractor2():
    model = FFNet(arch=[3, 3])
    loss_fn = nn.CrossEntropyLoss()
    me = ModelExtractor(model, loss_fn)
    x = next(iter(dl_tr))[0][0]
    if x.dtype is not torch.int64:
        res = me.get_decision_boundary(x, n_epochs=5)
        assert res.shape[-1] == 3

def test_extractor3():
    model = FFNet(arch=[3, 3])
    loss_fn = nn.CrossEntropyLoss()
    me = ModelExtractor(model, loss_fn)
    x = next(iter(dl_tr))[0][0]
    list_activations = me.get_activations(x)
    assert len(list_activations) == 2


