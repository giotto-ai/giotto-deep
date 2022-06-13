from typing import Dict, Union

from torch import nn
import torchvision.models as models
from torch.optim import SGD
from torch.utils.data.sampler import SubsetRandomSampler

from gdeep.models import FFNet
from gdeep.data.datasets import DatasetBuilder, DataLoaderBuilder
from gdeep.data.preprocessors import ToTensorImage
from gdeep.search import HyperParameterOptimization, GiottoSummaryWriter, Benchmark


# parametric model with string value
class model2(nn.Module):
    def __init__(self, n_nodes="100"):
        super(model2, self).__init__()
        self.md = nn.Sequential(nn.Sequential(models.resnet18(pretrained=True),
                                              nn.Linear(1000, eval(n_nodes))),
                                nn.Linear(eval(n_nodes), 10))

    def forward(self, x):
        return self.md(x)


models_dicts = []

model = model2()

temp_dict: Dict[str, Union[nn.Module, str]] = {"name": "resnet18", "model": model}

models_dicts.append(temp_dict)


# avoid having exposed parameters that wll not be searched on
class model_no_param(nn.Module):
    def __init__(self):
        super(model_no_param, self).__init__()
        self.mod = FFNet([3, 5, 5, 2])

    def forward(self, x):
        return self.mod(x)


model5 = model_no_param()
temp_dict = {"name": "ffnn", "model": model5}

models_dicts.append(temp_dict)


writer = GiottoSummaryWriter()

# download CIFAR10
bd = DatasetBuilder(name="CIFAR10")
ds_tr, _, _ = bd.build(download=True)

# Preprocessing
transformation = ToTensorImage([32, 32])
transformation.fit_to_dataset(ds_tr)  # this is useless for this transformation

transformed_ds_tr = transformation.attach_transform_to_dataset(ds_tr)

dataloaders_dicts = []


test_indices = [64*5 + x for x in range(32*3)]
train_indices = [x for x in range(32*2)]

dl = DataLoaderBuilder([transformed_ds_tr, transformed_ds_tr])
dl_tr, dl_val, _ = dl.build([{"batch_size": 32, "sampler": SubsetRandomSampler(train_indices)},  # type: ignore
                             {"batch_size": 32, "sampler": SubsetRandomSampler(test_indices)}])


temp_dict = {"name": "CIFAR10_1000", "dataloaders": [dl_tr, dl_val]}  # type: ignore

dataloaders_dicts.append(temp_dict)

db = DatasetBuilder(name="DoubleTori")
ds_tr, ds_val, _ = db.build()

dl_tr, dl_ts, _ = DataLoaderBuilder([ds_tr, ds_val]).build([{"batch_size": 48}, {"batch_size": 32}])  # type: ignore

temp_dict = {"name": "double_tori", "dataloaders": [dl_tr, dl_ts]}  # type: ignore

dataloaders_dicts.append(temp_dict)


def test_benchmark():
    # initialise loss
    loss_fn = nn.CrossEntropyLoss()

    # initialise the benchmarking class. When we do not specify it, it will use KFold with 5 splits
    bench = Benchmark(models_dicts, dataloaders_dicts, loss_fn, writer)

    # start the benchmarking
    bench.start(SGD, 1, True, {"lr": 0.01}, {"batch_size": 23})


def test_benchmark2():
    # initialise loss
    loss_fn = nn.CrossEntropyLoss()

    # initialise the benchmarking class. When we do not specify it, it will use KFold with 5 splits
    bench = Benchmark(models_dicts, dataloaders_dicts, loss_fn, writer)

    # start the benchmarking
    bench.start(SGD, 2, False, {"lr": 0.01}, {"batch_size": 32}, n_accumulated_grads=2)
