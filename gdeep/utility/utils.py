import torch


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
