import torch
from torch import nn
import os
import time


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
                             optimizer=None):
    """Save the model and the optimizer state_dict

    Args:
        model (nn.Module):
            the model to be saved
        model_name (str):
            model name
        optimizer (torch.optim):
            the optimizer to save
    """
    if os.path.exists("state_dicts"):
        pass
    else:
        os.makedirs("state_dicts")

    if model_name is None:
        torch.save(model.state_dict(),
                   os.path.join("state_dicts",
                                model.__class__.__name__+"-"+str(round(time.time()))+".pth"))
    else:
        torch.save(model.state_dict(),
                   os.path.join("state_dicts",
                                model_name +"-"+ str(round(time.time()))+".pth"))
    if optimizer is not None:
        torch.save(optimizer.state_dict(),
                   os.path.join("state_dicts",
                                str(optimizer).replace("\n","").replace("(","").replace(":","").replace(")","")
                                +"-"+ str(round(time.time()))+".pth"))


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