from ctypes import Union
import imp
from IPython import get_ipython  # type: ignore

import base64
import os
import time
import hashlib
import warnings
import torch
from torch import nn


def _are_compatible(model_dict, dataloaders_dict):
    """utility function to check the compatibility of a model
    with a set of dataloaders `(dl_tr, dl_val, dl_ts)`
    """

    model = model_dict["model"]
    batch = next(iter(dataloaders_dict["dataloaders"][0]))[0]
    try:
        model(batch)
    except RuntimeError:
        return False
    else:
        return True

def save_model_and_optimizer(model,
                             model_name: str=None,
                             trial_id: str=None,
                             optimizer=None,
                             store_pickle=False):
    """Save the model and the optimizer state_dict

    Args:
        model (nn.Module):
            the model to be saved
        model_name (str):
            model name
        trial_id (str):
            trial id to add to the name
        optimizer (torch.optim):
            the optimizer to save
        store_pickle (bool, default False):
            whether to store the pickle file of the model
            instead of the state_dict. The default
            is for state_dicts
    """

    if not trial_id:
        trial_id = str(round(time.time()))

    if os.path.exists("state_dicts"):
        pass
    else:
        os.makedirs("state_dicts")

    if model_name is None:
        if store_pickle:
            torch.save(model,
                       os.path.join("state_dicts",
                                    model.__class__.__name__ + "-" + trial_id + ".pickle"))
        else:
            torch.save(model.state_dict(),
                       os.path.join("state_dicts",
                                    model.__class__.__name__+"-"+trial_id+".pth"))
    else:
        if store_pickle:
            torch.save(model,
                       os.path.join("state_dicts",
                                    model_name + "-" + trial_id + ".pickle"))
        else:
            torch.save(model.state_dict(),
                       os.path.join("state_dicts",
                                    model_name +"-"+trial_id+".pth"))
    if optimizer is not None:
        torch.save(optimizer.state_dict(),
                   os.path.join("state_dicts",
                                str(optimizer).replace("\n","").replace("(","").replace(":","").replace(")","")
                                +"-"+trial_id+".pth"))


def ensemble_wrapper(clss):
    """function to wrap the ensemble estimators
    of the ``torchensable`` library.

    The only argument is the estimator class. Then
    you can initialise the output of this function
    as you would normally do for the original
    ``torchensebl``` class

    Args
        clss (type):
            the class of the estimator, like
            ``VotingClassifier`` for example
    """

    class NewEnsembleEstimator(clss):
        def __init__(self, *args, **kwargs):
            super(NewEnsembleEstimator, self).__init__(*args, **kwargs)
            self.estimators_ = nn.ModuleList().extend([self._make_estimator() for _ in range(self.n_estimators)])

    return NewEnsembleEstimator


def _inner_refactor_scalars(list_, cross_validation, k_folds):
    """used to restructure lists of accuracies and losses
    per epoch"""
    out = []
    for t in range(len(list_)):
        lis = [x[0] for x in list_ if x[1]==t]
        value = sum(lis)
        if len(lis) > 0:
            if cross_validation:
                out.append([value/k_folds , t])
            else:
                out.append([value, t])
    return out

def is_notebook() -> bool:
    """Check if the current environment is a notebook
    
    Returns:
        bool:
            True if the environment is a notebook, False otherwise
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
def autoreload_if_notebook() -> None:
    """Autoreload the modules if the environment is a notebook
    
    Returns:
        None
    """
    from IPython import get_ipython  # type: ignore
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
    
def _file_as_bytes(file) -> bytes:
    """Returns a bytes object representing the file

    Args:
        file (str):
            Path to the file

    Returns:
        bytes:
            Bytes object representing the file.
    """
    with open(file, 'rb') as f:
        return f.read()
    
def get_checksum(file: str, encoding: str = "hex"):
    """Returns the checksum of the file

    Args:
        file (str):
            Path to the file
            
    Raises:
        ValueError: if the file does not exist
        ValueError: if the encoding is not supported
        

    Returns:
            The checksum of the file. If the file does not exist,
            None is returned.
    """
    # Check if file exists
    if not os.path.exists(file):
        raise ValueError("File {} does not exist".format(file))
    if encoding == "hex":
        return hashlib.md5(_file_as_bytes(file)).hexdigest()
    elif encoding == "base64":
        return base64.b64encode(
            bytes.fromhex(hashlib.md5(_file_as_bytes(file)).hexdigest())
            )
    else:
        raise ValueError("encoding must be either 'hex' or 'base64'")


def flatten_list_of_lists(list_: list) -> list:
    """Flatten a list of lists

    Args:
        list_ (list):
            the list to flatten

    Returns:
        list:
            the flattened list
    """
    return [item for sublist in list_ for item in sublist]


class KnownWarningSilencer:
    """silence all warnings within this ``with``
    statement with this class"""

    def __init__(self):
        pass

    def __enter__(self):
        warnings.filterwarnings("ignore")
        return self

    def __exit__(self, type, value, traceback):
        warnings.filterwarnings("default")
