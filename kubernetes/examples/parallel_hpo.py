import socket
from datetime import datetime
import os

from torch import nn
from torch.optim import SGD
from sklearn.model_selection import StratifiedKFold

from gdeep.models import FFNet
from gdeep.trainer import Trainer
from gdeep.data.datasets import DatasetBuilder, DataLoaderBuilder
from gdeep.search import GiottoSummaryWriter
from gdeep.search import HyperParameterOptimization


# wrap all the hpo into a function with no arguments
def hpo_parallel(usr: str, psw: str, host: str, study_name: str) -> None:
    """The function containing all the gridsearch to execute
    Args:
        usr:
            the username for mySQL server
        psw:
            the password for the user
        host:
            IP of the host. Port is 3306
        study_name:
            the name of the optuna study
    """

    current_time: str = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(
        "giotto-deep", "runs", current_time + "_" + socket.gethostname()
    )
    writer = GiottoSummaryWriter(log_dir=log_dir)

    bd = DatasetBuilder(name="DoubleTori")
    ds_tr, ds_val, _ = bd.build()

    dl_builder = DataLoaderBuilder([ds_tr, ds_val])  # type: ignore
    dl_tr, dl_val, dl_ts = dl_builder.build([{"batch_size": 23}, {"batch_size": 23}])

    # build the model
    class Model1(nn.Module):
        def __init__(self):
            super(Model1, self).__init__()
            self.seqmodel = nn.Sequential(nn.Flatten(), FFNet(arch=(3, 5, 10, 5, 2)))

        def forward(self, x):
            return self.seqmodel(x)

    model = Model1()

    # initlaise the loss function
    loss_fn = nn.CrossEntropyLoss()

    # initialise the pipelien class
    pipe = Trainer(
        model,
        [dl_tr, dl_val],
        loss_fn,
        writer,
        k_fold_class=StratifiedKFold(3, shuffle=True),
    )

    # initialise the SAM optimiser
    optim = SGD  # this is a class, not an instance!

    search = HyperParameterOptimization(
        pipe,
        "loss",
        20,
        study_name=study_name,
        db_url="mysql+mysqldb://" + usr + ":" + psw + "@" + host + ":3306/example",
    )
    # dictionaries of hyperparameters
    optimizers_params = {"lr": [0.001, 0.01]}
    dataloaders_params = {"batch_size": [32, 64, 16]}

    # starting the HPO
    search.start(
        [optim],  # type: ignore
        5,
        False,
        optimizers_params,
        dataloaders_params,
        n_accumulated_grads=2,
    )
