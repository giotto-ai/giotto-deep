from gdeep.pipeline import Pipeline
from gdeep.utility import _are_compatible

class Benchmark:
    """This is the generic class that allows
    the user to perform benchmarking over different
    datasets and models.

    Args:
        models_dicts (list of dicts):
            each dictionary has two items, `"name":"string"`
            the name of the model and `"model":nn.Module` a
            standard torch model
        dataloaders_dicts (utils.DataLoader):
            each dictionary has two items, `"name":"string"`
            the name of the model and `"dataloaders":list` a
            list of standard torch.dataloaders, e.g. `(dl_tr, dl_ts)`
        loss_fn (Callables):
            loss function
        wirter (tensorboard SummaryWriter):
            tensorboard writer

    """

    def __init__(self, models_dicts, dataloaders_dicts, loss_fn, writer):
        self.models_dicts = models_dicts
        self.dataloaders_dicts = dataloaders_dicts
        self.loss_fn = loss_fn
        self.writer = writer
        if not isinstance(self.models_dicts, list):
            raise TypeError("The provided models must be a Python list of dictionaries")

        if not isinstance(self.dataloaders_dicts, list):
            raise TypeError("The provided datasets must be a Python list of dictionaries")

    def start(self, optimizer,
              n_epochs=10,
              cross_validation=False,
              optimizer_param=None,
              dataloaders_param=None,
              lr_scheduler=None,
              scheduler_params=None,
              profiling=False,
              k_folds=5):
        """Method to be called when starting the benchmarking
        
        Args:
            optimizer (torch.optim):
                a torch optimizers
            n_epochs (int):
                number of training epochs
            crossvalidation (bool):
                whether or not to use cross-validation
            optimizers_param (dict):
                dictionary of the optimizers
                parameters, e.g. `{"lr": 0.001}`
            dataloaders_param (dict):
                dictionary of the dataloaders
                parameters, e.g. `{"batch_size": 32}`
            lr_scheduler (torch.optim):
                a learning rate scheduler
            scheduler_params (dict):
                learning rate scheduler parameters
            profiling (bool, default=False):
                whether or not you want to activate the
                profiler
            k_folds (int, default=5):
                number of folds in cross validation
        """

        print("Benchmarking Started")
        for dataloaders in self.dataloaders_dicts:
            for model in self.models_dicts:
                if _are_compatible(model, dataloaders):
                    print("*"*40)
                    print("Training on Dataset: {}, Model: {}".format(dataloaders["name"], model["name"]))
                    pipe = Pipeline(model["model"], dataloaders["dataloaders"], self.loss_fn, self.writer)
                    pipe.train(optimizer, n_epochs,
                               cross_validation,
                               optimizer_param,
                               dataloaders_param,
                               lr_scheduler,
                               scheduler_params,
                               None,
                               profiling,
                               k_folds)
