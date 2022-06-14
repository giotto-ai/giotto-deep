import warnings
from typing import Tuple, Any, Dict, Callable, Type, Optional, List, Union

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection._split import BaseCrossValidator  # noqa
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler  # noqa
from optuna.trial._base import BaseTrial  # noqa
from sklearn.model_selection import KFold

from gdeep.trainer import Trainer
from gdeep.utility import _are_compatible  # noqa
from gdeep.trainer import accuracy, TrainerConfig

Tensor = torch.Tensor


class Benchmark:
    """This is the generic class that allows
    the user to perform benchmarking over different
    datasets and models.

    Args:
        models_dicts :
            each dictionary has two items, ``"name":"string"``
            the name of the model and `"model":nn.Module` a
            standard torch model
        dataloaders_dicts :
            each dictionary has two items, ``"name":"string"``
            the name of the model and ``"dataloaders":list`` a
            list of standard torch.dataloaders, e.g. ``(dl_tr, dl_ts)``
        loss_fn:
            loss function
        writer:
            tensorboard writer
        training_metric:
            the function that computes the metric: it shall
            have two arguments, one for the prediction
            and the other for the ground truth
        k_fold_class:
            the class instance to implement the KFold, can be
            any of the Splitter classes of sklearn. More
            info at https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

    """

    def __init__(
        self,
        models_dicts: Dict[str, Union[torch.nn.Module, str]],
        dataloaders_dicts: Dict[
            str, Union[List[DataLoader[Tuple[Tensor, Tensor]]], str]
        ],
        loss_fn: Callable[[Tuple[Tensor, Tensor]], Tensor],
        writer: SummaryWriter,
        training_metric: Optional[Callable[[Tensor, Tensor], float]] = None,
        k_fold_class: Optional[BaseCrossValidator] = None,
    ) -> None:
        self.models_dicts = models_dicts
        self.dataloaders_dicts = dataloaders_dicts
        self.loss_fn = loss_fn
        self.writer = writer
        if not self.writer:
            warnings.warn("No writer detected")

        if not k_fold_class:
            self.k_fold_class = KFold(5, shuffle=True)
        else:
            self.k_fold_class = k_fold_class
        if training_metric:
            self.training_metric = training_metric
        else:
            self.training_metric = accuracy
        if not isinstance(self.models_dicts, list):
            raise TypeError("The provided models must be a Python list of dictionaries")

        if not isinstance(self.dataloaders_dicts, list):
            raise TypeError(
                "The provided datasets must be a Python list of dictionaries"
            )

    def start(
        self,
        optimizer: Type[Optimizer],
        n_epochs: int = 10,
        cross_validation: bool = False,
        optimizers_param: Optional[Dict[str, Any]] = None,
        dataloaders_param: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        profiling: bool = False,
        parallel_tpu: bool = False,
        keep_training: bool = False,
        store_grad_layer_hist: bool = False,
        n_accumulated_grads: int = 0,
        writer_tag: str = "",
    ) -> None:
        """Method to be called when starting the benchmarking
        
        Args:
            optimizer:
                a torch optimizers class (not the instance)
            n_epochs:
                number of training epochs
            cross_validation:
                whether or not to use cross-validation
            optimizers_param:
                dictionary of the optimizers
                parameters, e.g. ``{"lr": 0.001}``
            dataloaders_param:
                dictionary of the dataloaders
                parameters, e.g. ``{"batch_size": 32}``
            lr_scheduler:
                a learning rate scheduler class (not instance)
            scheduler_params:
                learning rate scheduler parameters
            profiling:
                whether or not you want to activate the
                profiler
            parallel_tpu:
                boolean value to run the computations
                on multiple TPUs
            keep_training:
                This flag allows to restart a training from
                the existing optimizer as well as the
                existing model
            store_grad_layer_hist:
                This flag allows to store the gradients
                and the layer values in tensorboard for
                each epoch
            n_accumulated_grads:
                number of accumulated gradients. It is
                considered only if a positive integer
            writer_tag:
                string to be added to the tensorboard items title
        """

        print("Benchmarking Started")
        config = TrainerConfig(
            optimizer,
            n_epochs,
            cross_validation,
            optimizers_param,
            dataloaders_param,
            lr_scheduler,
            scheduler_params,
            None,
            profiling,
            parallel_tpu,
            keep_training,
            store_grad_layer_hist,
            n_accumulated_grads,
            writer_tag,
        )

        _benchmarking_param(
            self._inner_function, (self.models_dicts, self.dataloaders_dicts), config
        )

    def _inner_function(
        self,
        model: Dict[str, Union[torch.nn.Module, str]],
        dataloaders: Dict[str, Union[List[DataLoader[Tuple[Tensor, Tensor]]], str]],
        config: TrainerConfig,
    ) -> None:
        """private method to run the inner
        function of the benchmark loops
        
        Args:
            model:
                dictionary defining the model name
                and actual nn.Module
            dataloaders:
                dictionary defining the dataset name
                and the actual list of dataloaders, e.g.
                ``[dl_tr, dl_val, dl_ts]``
            config:
                the configuration class ``TrainerConfig``.
        """
        pipe = Trainer(
            model["model"],
            dataloaders["dataloaders"],
            self.loss_fn,
            self.writer,
            self.training_metric,  # type: ignore
            self.k_fold_class,
        )
        config.writer_tag += "Dataset:" + dataloaders["name"] + "|Model:" + model["name"]  # type: ignore
        pipe.train(**config.to_dict())


def _benchmarking_param(
    fun: Callable[[Any], Any],
    arguments: Tuple[
        Dict[str, Union[torch.nn.Module, str]],
        Dict[str, Union[List[DataLoader[Tuple[Tensor, Tensor]]], str]],
    ],
    config: TrainerConfig,
) -> None:
    """Function to be used as pseudo-decorator for
    benchmarking loops
    
    Args:
        fun (Callable):
            the function to decorate
        arguments (list):
            list of arguments to pass to the inner
            function of the wrapper. Expected to receive
            ``models_dicts, dataloaders_dict = arguments``
        config:
            the configuration class ``TrainerConfig``.

    """

    models_dicts, dataloaders_dicts = arguments
    for dataloaders in dataloaders_dicts:
        for model in models_dicts:
            if _are_compatible(model, dataloaders):
                print("*" * 40)
                print(
                    f"Performing Gridsearch on Dataset: {dataloaders['name']}"
                    f", Model: {model['name']}"
                )
                fun(model, dataloaders, config)
