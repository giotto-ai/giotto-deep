from dataclasses import dataclass
from typing import List, Any, Type, Optional, Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler  # noqa


@dataclass
class HPOConfig:
    """Config class for thee HPO parameters

    Args:
        optimizers:
            list of torch optimizers classes, not isntances
        n_epochs:
            number of training epochs
        cross_validation:
            whether or not to use cross-validation
        optimizers_params:
            dictionary of optimizers params
        dataloaders_params:
            dictionary of dataloaders parameters
        models_hyperparams:
            dictionary of model parameters
        lr_scheduler:
            torch learning rate schduler class
        schedulers_params:
            learning rate scheduler parameters
        profiling :
            whether or not you want to activate the
            profiler
        parallel_tpu:
            boolean value to run the computations
            on multiple TPUs
        n_accumulated_grads (int, default=0):
            number of accumulated gradients. It is
            considered only if a positive integer
        keep_training:
            bool flag to decide whether to continue
            training or not
        store_grad_layer_hist:
            flag to store the gradents of the layers in the
            tensorboard histograms
        writer_tag:
            tag to prepend to the ouput
            on tensorboard"""

    optimizers: List[Type[Optimizer]]
    n_epochs: int = 1
    cross_validation: bool = False
    optimizers_params: Optional[Dict[str, Any]] = None
    dataloaders_params: Optional[Dict[str, Any]] = None
    models_hyperparams: Optional[Dict[str, Any]] = None
    lr_scheduler: Optional[Type[_LRScheduler]] = None
    schedulers_params: Optional[Dict[str, Any]] = None
    profiling: bool = False
    parallel_tpu: bool = False
    keep_training: bool = False
    store_grad_layer_hist: bool = False
    n_accumulated_grads: int = 0
    writer_tag: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """method to transform the config file into a dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
