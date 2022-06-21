import sys

from torch import nn
from torch.optim import SGD
from sklearn.model_selection import StratifiedKFold

from gdeep.models import FFNet
from gdeep.utility.optimisation import SAMOptimizer
from gdeep.trainer import Trainer
from gdeep.data.datasets import DatasetBuilder, DataLoaderBuilder
from gdeep.search import GiottoSummaryWriter
from gdeep.search import HyperParameterOptimization


def install(package):
    """function to install a package, like
    ``'mysql-connector-python'``"""
    print(f"Installing the package {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def connect_to_mysql(usr: str, psw: str, host: str) -> None:
    """function to connect to mysql database

    Args:
        usr:
            the username for mySQL server
        psw:
            the password for the user
        host:
            IP of the host. Port is 3306
    """
    import mysql.connector
    mydb = mysql.connector.connect(
        host=host,
        user=usr,
        password=psw
    )

    print(mydb)
    mycursor = mydb.cursor()

    mycursor.execute("CREATE DATABASE IF NOT EXISTS example")

    mycursor.execute("SHOW DATABASES")

    for x in mycursor:
        print(x)


# testing the working of RQ
def test_fnc(text: str) -> int:
    """testing function

    Args:
        text:
            a string

    Returns:
        int:
            the length of the input string
    """
    return len(text)


# wrap all the hpo into a function with no arguments
def run_hpo_parallel(usr: str, psw: str, host: str) -> None:
    """The function containing all the gridsearch to execute
    Args:
        usr:
            the username for mySQL server
        psw:
            the password for the user
        host:
            IP of the host. Port is 3306
    """
    writer = GiottoSummaryWriter()

    bd = DatasetBuilder(name="DoubleTori")
    ds_tr, ds_val, _ = bd.build()

    dl_builder = DataLoaderBuilder((ds_tr, ds_val))
    dl_tr, dl_val, dl_ts = dl_builder.build(({"batch_size": 23}, {"batch_size": 23}))

    # build the model
    class Model1(nn.Module):
        def __init__(self):
            super(Model1, self).__init__()
            self.seqmodel = nn.Sequential(nn.Flatten(), FFNet(arch=[3, 5, 10, 5, 2]))

        def forward(self, x):
            return self.seqmodel(x)

    model = Model1()

    # initlaise the loss function
    loss_fn = nn.CrossEntropyLoss()

    # initialise the pipelien class
    pipe = Trainer(
        model,
        (dl_tr, dl_val),
        loss_fn,
        writer,
        k_fold_class=StratifiedKFold(3, shuffle=True),
    )

    # initialise the SAM optimiser
    optim = SAMOptimizer(SGD)  # this is a class, not an instance!

    search = HyperParameterOptimization(pipe, "loss", 20, study_name="distributed-example-6",
                                        db_url="mysql+mysqldb://"+usr+":"+psw+"@"+host+":3306/example")
    # dictionaries of hyperparameters
    optimizers_params = {"lr": [0.001, 0.01]}
    dataloaders_params = {"batch_size": [32, 64, 16]}
    models_hyperparams = {"n_nodes": ["200"]}

    # starting the HPO
    search.start(
        [optim],
        5,
        False,
        optimizers_params,
        dataloaders_params,
        models_hyperparams,
        n_accumulated_grads=2,
    )
