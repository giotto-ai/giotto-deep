from gdeep.pipeline import Pipeline
from gdeep.utility import _are_compatible
import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

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
              k_folds=5,
              parallel_tpu=False):
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
            parallel_tpu (bool):
                boolean value to run the computations
                on multiple TPUs
        """

        print("Benchmarking Started")
        _benchmarking_param(self._inner_function,
                            [self.models_dicts,
                             self.dataloaders_dicts],
                            optimizer, n_epochs,
                            cross_validation,
                            optimizer_param,
                            dataloaders_param,
                            lr_scheduler,
                            scheduler_params,
                            profiling,
                            k_folds,
                            parallel_tpu,
                            writer_tag="")


    def _inner_function(self, model,
                        dataloaders,
                        optimizer, n_epochs,
                        cross_validation,
                        optimizer_param,
                        dataloaders_param,
                        lr_scheduler,
                        scheduler_params,
                        profiling,
                        k_folds,
                        parallel_tpu,
                        writer_tag=""):
        """private method to run the inner
        function of the benchmark loops
        
        Args:
            model (dict):
                dictionary defining the model name
                and actual nn.Module
            dataloaders (dict):
                dictionary defining the dataset name
                and the actual list of dataloaders, e.g.
                ``[dl_tr, dl_val, dl_ts]``
            optimizer (torch.optim):
                a torch optimizers
            n_epochs (int):
                number of training epochs
            cross_validation (bool):
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
            parallel_tpu (bool):
                boolean value to run the computations
                on multiple TPUs
            writer_tag (str):
                the tensorboard writer tag
        """
        pipe = Pipeline(model["model"], dataloaders["dataloaders"],
                        self.loss_fn, self.writer)
        writer_tag += "Dataset:" + dataloaders["name"] +"|Model:" + model["name"]
        pipe.train(optimizer, n_epochs,
                   cross_validation,
                   optimizer_param,
                   dataloaders_param,
                   lr_scheduler,
                   scheduler_params,
                   None,
                   profiling,
                   k_folds,
                   parallel_tpu,
                   writer_tag)

def _benchmarking_param(fun, arguments, *args, **kwargs):
    """Function to be used as pseudo-decorator for
    benchmarking loops
    
    Args:
        fun (Callable):
            the function to decorate
        arguments (list):
            list of arguments to pass to the inner
            function of the wrapper. Expected to receive
            ``models_dicts, dataloaders_dict = arguments``
        *args (*list):
            all the args of ``fun``
        **kwargs (**dict):
            all the kwargs of ``fun``

    """

    models_dicts, dataloaders_dicts = arguments
    for dataloaders in dataloaders_dicts:
        for model in models_dicts:
            if _are_compatible(model, dataloaders):
                print("*"*40)
                print("Performing Gridsearch on Dataset: {}, Model: {}".format(dataloaders["name"],
                                                                               model["name"]))
                fun(model, dataloaders, *args, **kwargs)

