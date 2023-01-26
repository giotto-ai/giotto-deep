from dataclasses import dataclass
from typing import Type, Optional, Dict, Any, Tuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler  # noqa
from optuna.trial._base import BaseTrial  # noqa


@dataclass
class TrainerConfig:
    """Configuration class that contains the
    parameters of the Trainer and of the Benchmark class

    Args:
        optimizer:
            optimizer class, not the instance
        n_epochs:
            number of training epochs
        optimizers_params:
            dictionary of the optimizers
            parameters, e.g. `{"lr": 0.001}`
        dataloaders_params:
            dictionary of the dataloaders
            parameters
        models_hyperparams:
            dictionary of the model
            parameters
        lr_scheduler:
            a learning rate scheduler class
        schedulers_params:
            learning rate scheduler parameters
        optuna_params:
            a tuple with the optuna trial and the
            search metric (a string).
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
            this is the number of accumulated grads. It
            is taken into account only for positive integers
        writer_tag:
            tag to prepend to the output
            on tensorboard
    """

    optimizer: Type[Optimizer]
    n_epochs: int = 10
    cross_validation: bool = False
    optimizers_param: Optional[Dict[str, Any]] = None
    dataloaders_param: Optional[Dict[str, Any]] = None
    lr_scheduler: Optional[Type[_LRScheduler]] = None
    scheduler_params: Optional[Dict[str, Any]] = None
    optuna_params: Optional[Tuple[BaseTrial, str]] = None
    profiling: bool = False
    parallel_tpu: bool = False
    keep_training: bool = False
    store_grad_layer_hist: bool = False
    n_accumulated_grads: int = 0
    writer_tag: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """method to transform the config file into a dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
