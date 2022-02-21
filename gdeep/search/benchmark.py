from gdeep.pipeline import Pipeline
from gdeep.utility import _are_compatible
import torch
from sklearn.model_selection import KFold
import warnings

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
        writer (tensorboard SummaryWriter):
            tensorboard writer
        KFold_class (sklearn.model_selection, default KFold()):
            the class instance to implement the KFold, can be
            any of the Splitter classes of sklearn. More
            info at https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

    """

    def __init__(self, models_dicts, dataloaders_dicts, loss_fn, writer, KFold_class=None):
        self.models_dicts = models_dicts
        self.dataloaders_dicts = dataloaders_dicts
        self.loss_fn = loss_fn
        self.writer = writer
        if not self.writer:
            warnings.warn("No writer detected")
        
        if not KFold_class:
            self.KFold_class = KFold(5, shuffle=True)
        else:
            self.KFold_class = KFold_class

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
              parallel_tpu=False,
              keep_training=False,
              store_grad_layer_hist=False,
              n_accumulated_grads=0):
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
            parallel_tpu (bool):
                boolean value to run the computations
                on multiple TPUs
            keep_training (bool):
                This flag allows to restart a training from
                the existing optimizer as well as the
                existing model
            store_grad_layer_hist (bool):
                This flag allows to store the gradients
                and the layer values in tensorboard for
                each epoch
            n_accumulated_grads (int, default=0):
                number of accumulated gradients. It is
                considered only if a positive integer
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
                            parallel_tpu,
                            keep_training,
                            store_grad_layer_hist,
                            n_accumulated_grads,
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
                        parallel_tpu,
                        keep_training,
                        store_grad_layer_hist,
                        n_accumulated_grads,
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
            parallel_tpu (bool):
                boolean value to run the computations
                on multiple TPUs
            keep_training (bool):
                boolean flag to use the same model for
                further training
            store_grad_layer_hist (bool):
                flag to store or not the layer's grads
                on tensorboard
            n_accumulated_grads (int):
                this is the number of accumated grads. It
                is taken into account only for positive integers
            writer_tag (str):
                the tensorboard writer tag
        """
        pipe = Pipeline(model["model"], dataloaders["dataloaders"],
                        self.loss_fn, self.writer, self.KFold_class)
        writer_tag += "Dataset:" + dataloaders["name"] +"|Model:" + model["name"]
        pipe.train(optimizer, n_epochs,
                   cross_validation,
                   optimizer_param,
                   dataloaders_param,
                   lr_scheduler,
                   scheduler_params,
                   None,
                   profiling,
                   parallel_tpu,
                   keep_training,
                   store_grad_layer_hist,
                   n_accumulated_grads,
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
                print(f"Performing Gridsearch on Dataset: {dataloaders['name']}"
                      f", Model: {model['name']}")
                fun(model, dataloaders, *args, **kwargs)

