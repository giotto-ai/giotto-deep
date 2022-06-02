from torch import nn
import torchvision.models as models
from torch.optim import SGD, Adam
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler

from gdeep.trainer import Trainer
from gdeep.data.datasets import DatasetBuilder, DataLoaderBuilder
from gdeep.data.preprocessors import ToTensorImage
from gdeep.search import HyperParameterOptimization, GiottoSummaryWriter


# parametric model with string value
class Model2(nn.Module):
    def __init__(self, n_nodes="100"):
        super(Model2, self).__init__()
        self.md = nn.Sequential(nn.Sequential(models.resnet18(pretrained=True),
                                              nn.Linear(1000, eval(n_nodes))),
                                nn.Linear(eval(n_nodes), 10))

    def forward(self, x):
        return self.md(x)


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
dl_tr, *_ = DataLoaderBuilder([transformed_ds_tr]).build(
    [{"batch_size": 32, "sampler": SubsetRandomSampler(train_indices)}])


def test_hpo_failure():
    # define the model
    model = Model2()

    # initialise loss
    loss_fn = nn.CrossEntropyLoss()

    # initialise pipeline class
    pipe = Trainer(model, [dl_tr, None], loss_fn, writer,
                   k_fold_class=StratifiedKFold(2, shuffle=True))

    # initialise gridsearch
    try:
        HyperParameterOptimization(pipe, "accu", 2, best_not_last=True)
    except AssertionError:
        pass

def test_hpo_cross_val():
    # define the model
    model = Model2()

    # initialise loss
    loss_fn = nn.CrossEntropyLoss()

    # initialise pipeline class
    pipe = Trainer(model, [dl_tr, None], loss_fn, writer,
                   k_fold_class=StratifiedKFold(2, shuffle=True))

    # initialise gridsearch
    search = HyperParameterOptimization(pipe, "accuracy", 2, best_not_last=True)

    # if you want to store pickle files of the models instead of the state_dicts
    search.store_pickle = True

    # dictionaries of hyperparameters
    optimizers_params = {"lr": [0.001, 0.01]}
    dataloaders_params = {"batch_size": [32, 64, 16]}
    models_hyperparams = {"n_nodes": ["200"]}

    # starting the gridsearch
    search.start([SGD, Adam], 3, False, optimizers_params, dataloaders_params,
                 models_hyperparams, n_accumulated_grads=2)


def test_hpo_accumulated_grads():
    # define the model
    model = Model2()

    # initialise loss
    loss_fn = nn.CrossEntropyLoss()

    # initialise pipeline class
    pipe = Trainer(model, [dl_tr, None], loss_fn, writer)

    # initialise gridsearch
    search = HyperParameterOptimization(pipe, "accuracy", 2, best_not_last=True)

    # if you want to store pickle files of the models instead of the state_dicts
    search.store_pickle = True

    # dictionaries of hyperparameters
    optimizers_params = {"lr": [0.001, 0.01]}
    dataloaders_params = {"batch_size": [32, 64, 16]}
    models_hyperparams = {"n_nodes": ["200"]}

    # starting the gridsearch
    search.start([SGD, Adam], 2, False, optimizers_params, dataloaders_params,
                 models_hyperparams, n_accumulated_grads=2)


def test_hpo_loss():
    # define the model
    model = Model2()

    # initialise loss
    loss_fn = nn.CrossEntropyLoss()

    # initialise pipeline class
    pipe = Trainer(model, [dl_tr, None], loss_fn, writer)

    # initialise gridsearch
    search = HyperParameterOptimization(pipe, "loss", 2, best_not_last=True)

    # dictionaries of hyperparameters
    optimizers_params = {"lr": [0.001, 0.01]}
    dataloaders_params = {"batch_size": [32, 64, 16]}
    models_hyperparams = {"n_nodes": ["200"]}

    # starting the gridsearch
    search.start([SGD, Adam], 2, False, optimizers_params, dataloaders_params, models_hyperparams)


def test_hpo_string_parameters():
    # define the model
    model = Model2()

    # initialise loss
    loss_fn = nn.CrossEntropyLoss()

    # initialise pipeline class
    pipe = Trainer(model, [dl_tr, None], loss_fn, writer)

    # initialise gridsearch
    search = HyperParameterOptimization(pipe, "loss", 2)

    # if you want to store pickle files of the models instead of the state_dicts
    search.store_pickle = True

    # dictionaries of hyperparameters
    optimizers_params = {"lr": [0.001, 0.01]}
    dataloaders_params = {"batch_size": [32, 64]}
    models_hyperparams = {"n_nodes": ["200", "256"]}

    # starting the gridsearch
    search.start([SGD], 1, False, optimizers_params, dataloaders_params, models_hyperparams)
