import torch
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
                                "optimizer" +"-"+ str(round(time.time()))+".pth"))