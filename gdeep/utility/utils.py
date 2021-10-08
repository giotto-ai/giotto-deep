import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    DEVICE = xm.xla_device()
    print("Using TPU!")
except:
    print("No TPUs...")

def _are_compatible(model_dict, dataloaders_dict):
    """utility function to check the compatibility of a model
    with a set of dataloaders `(dl_tr, dl_val, dl_ts)`
    """

    model = model_dict["model"].to(DEVICE)
    batch = next(iter(dataloaders_dict["dataloaders"][0]))[0].to(DEVICE)
    try:
        model(batch)
    except RuntimeError:
        return False
    else:
        return True
