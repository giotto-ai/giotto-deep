from typing import List, Tuple

from torch import nn
import torchvision.models as models
from torch.optim import SGD, Adam
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

from gdeep.trainer import Trainer
from gdeep.data.datasets import DatasetBuilder, DataLoaderBuilder
from gdeep.data.preprocessors import ToTensorImage
from gdeep.search import HyperParameterOptimization, GiottoSummaryWriter
from gdeep.models import FFNet
from gdeep.trainer.regularizer import TihonovRegularizer

from gdeep.utility.custom_types import Tensor


# parametric model with string value
class Model2(nn.Module):
    def __init__(self, n_nodes="100"):
        super(Model2, self).__init__()
        self.md = nn.Sequential(
            nn.Sequential(
                models.resnet18(pretrained=True), nn.Linear(1000, eval(n_nodes))
            ),
            nn.Linear(eval(n_nodes), 10),
        )

    def forward(self, x):
        return self.md(x)


# model
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.seqmodel = nn.Sequential(nn.Flatten(), FFNet(arch=(4, 5, 4)))

    def forward(self, x):
        return self.seqmodel(x).reshape(-1, 2, 2)


writer = GiottoSummaryWriter()

# download MNIST
bd = DatasetBuilder(name="CIFAR10")
ds_tr, _, _ = bd.build(download=True)

# Preprocessing
transformation = ToTensorImage([32, 32])
transformation.fit_to_dataset(ds_tr)  # this is useless for this transformation

transformed_ds_tr = transformation.attach_transform_to_dataset(ds_tr)

# use only 32*7 images from CIFAR10
train_indices = list(range(32 * 7))
dl_tr, *_ = DataLoaderBuilder([transformed_ds_tr]).build(  # type: ignore
    [{"batch_size": 32, "sampler": SubsetRandomSampler(train_indices)}]
)


def test_hpo_failure():
    # define the model
    model = Model2()

    # initialise loss
    loss_fn = nn.CrossEntropyLoss()

    # initialise pipeline class
    pipe = Trainer(
        model,
        [dl_tr, None],  # type: ignore
        loss_fn,
        writer,
        k_fold_class=StratifiedKFold(2, shuffle=True),
    )

    # initialise gridsearch
    try:
        HyperParameterOptimization(pipe, "accuracy", 2, best_not_last=True)
    except AssertionError:
        pass


def test_hpo_cross_val():
    # define the model
    model = Model2()

    # initialise loss
    loss_fn = nn.CrossEntropyLoss()

    # initialise pipeline class
    pipe = Trainer(
        model,
        [dl_tr, None],  # type: ignore
        loss_fn,
        writer,
        k_fold_class=StratifiedKFold(2, shuffle=True),
    )

    # initialise gridsearch
    search = HyperParameterOptimization(pipe, "accuracy", 2, best_not_last=True)

    # if you want to store pickle files of the models instead of the state_dicts
    search.store_pickle = True

    # dictionaries of hyperparameters
    optimizers_params = {"lr": [0.001, 0.01]}
    dataloaders_params = {"batch_size": [32, 64, 16]}
    models_hyperparams = {"n_nodes": ["200"]}

    # starting the gridsearch
    search.start(
        [SGD, Adam],
        3,
        False,
        optimizers_params,
        dataloaders_params,
        models_hyperparams,
        n_accumulated_grads=2,
    )


def test_hpo_accumulated_grads():
    # define the model
    model = Model2()

    # initialise loss
    loss_fn = nn.CrossEntropyLoss()

    # initialise pipeline class
    pipe = Trainer(model, [dl_tr, None], loss_fn, writer)  # type: ignore

    # initialise gridsearch
    search = HyperParameterOptimization(pipe, "accuracy", 2, best_not_last=True)

    # if you want to store pickle files of the models instead of the state_dicts
    search.store_pickle = True

    # dictionaries of hyperparameters
    optimizers_params = {"lr": [0.001, 0.01]}
    dataloaders_params = {"batch_size": [32, 64, 16]}
    models_hyperparams = {"n_nodes": ["200"]}

    # starting the gridsearch
    search.start(
        [SGD, Adam],
        2,
        False,
        optimizers_params,
        dataloaders_params,
        models_hyperparams,
        n_accumulated_grads=2,
    )


def test_hpo_loss():
    # define the model
    model = Model2()

    # initialise loss
    loss_fn = nn.CrossEntropyLoss()

    # initialise pipeline class
    pipe = Trainer(model, [dl_tr, None], loss_fn, writer)  # type: ignore

    # initialise gridsearch
    search = HyperParameterOptimization(pipe, "loss", 2, best_not_last=True)

    # dictionaries of hyperparameters
    optimizers_params = {"lr": [0.001, 0.01]}
    dataloaders_params = {"batch_size": [32, 64, 16]}
    models_hyperparams = {"n_nodes": ["200"]}

    # starting the gridsearch
    search.start(
        [SGD, Adam], 2, False, optimizers_params, dataloaders_params, models_hyperparams
    )


def test_hpo_string_parameters():
    # define the model
    model = Model2()

    # initialise loss
    loss_fn = nn.CrossEntropyLoss()

    # initialise pipeline class
    pipe = Trainer(model, [dl_tr, None], loss_fn, writer)  # type: ignore

    # initialise gridsearch
    search = HyperParameterOptimization(pipe, "loss", 2)

    # if you want to store pickle files of the models instead of the state_dicts
    search.store_pickle = True

    # dictionaries of hyperparameters
    optimizers_params = {"lr": [0.001, 0.01]}
    dataloaders_params = {"batch_size": [32, 64]}
    models_hyperparams = {"n_nodes": ["200", "256"]}

    # starting the gridsearch
    search.start(
        [SGD], 1, False, optimizers_params, dataloaders_params, models_hyperparams
    )


def test_hpo_collate():
    class MyDataset(Dataset):
        def __init__(self):
            self.x = []
            for _ in range(100):
                self.x.append((torch.rand(1, np.random.randint(2, 4))).to(torch.float))
            self.y = np.array(
                np.random.randint(2, size=100 * 2).reshape(-1, 2), dtype=np.int64
            )

        def __len__(self):
            return 100

        def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
            return self.x[item], self.y[item]

    def collate_fn(batch_tuple: List):
        # print(batch_tuple)
        target = torch.zeros(len(batch_tuple), 4)
        label = torch.zeros(len(batch_tuple), 2).to(torch.long)
        for i, batch in enumerate(batch_tuple):
            source = batch[0]
            target[i, : len(source[-1])] = source
            label[i] = torch.tensor(batch[1]).to(torch.long)
        return target, label

    model = Model1()
    # dataloaders
    ds = MyDataset()
    dl_train = DataLoader(ds, batch_size=6, collate_fn=collate_fn)

    # loss function
    loss_fn = nn.CrossEntropyLoss()
    # pipeline
    pipe = Trainer(model, [dl_train, None], loss_fn, writer)  # type: ignore
    # initialise gridsearch
    search = HyperParameterOptimization(pipe, "loss", 2, best_not_last=True)

    # dictionaries of hyperparameters
    optimizers_params = {"lr": [0.001, 0.01]}
    dataloaders_params = {"batch_size": [3, 9, 2], "collate_fn": [collate_fn]}

    # starting the gridsearch
    search.start([SGD, Adam], 1, False, optimizers_params, dataloaders_params)


def test_regularizer_optimization():
    """
    Test to verify the regularizer is passed to the model when one is provided
    """
    # define the model
    model = Model2()

    # initialise loss
    loss_fn = nn.CrossEntropyLoss()

    # initialise pipeline class
    pipe = Trainer(model, [dl_tr, None], loss_fn, writer)  # type: ignore

    # initialise gridsearch
    search = HyperParameterOptimization(pipe, "loss", 2, best_not_last=True)

    # dictionaries of hyperparameters
    optimizers_params = {"lr": [0.001, 0.01]}
    dataloaders_params = {"batch_size": [32, 64, 16]}
    models_hyperparams = {"n_nodes": ["200"]}
    regularization_params = {
        "regularizer": [TihonovRegularizer],
        "lamda": [0.05, 0.5, 0.01],
        "p": [1],
    }
    # starting the gridsearch
    search.start(
        [SGD, Adam],
        2,
        False,
        optimizers_params,
        dataloaders_params,
        models_hyperparams,
        regularization_params=regularization_params,
    )
    search.pipe.regularizer
