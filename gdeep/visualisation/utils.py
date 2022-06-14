import os

from PIL import Image
import numpy as np
from plotly.io import write_image
import torch


def plotly2tensor(fig) -> torch.Tensor:
    """convert Plotly fig to an array.

    Args:
        fig (plotly GraphicObject):
            figure to convert to tensor

    Returns:
        Tensor:
            the tensor discretisation of the
            figure
    """
    try:
        write_image(fig, "deleteme.jpeg", format="jpeg", engine="orca")
    except ValueError:
        write_image(fig, "deleteme.jpeg", format="jpeg")
    img = Image.open("deleteme.jpeg")
    arr = np.asarray(img).copy()
    os.remove("deleteme.jpeg")
    return torch.from_numpy(arr)


def png2tensor(file_name) -> torch.Tensor:
    img = Image.open(file_name)
    arr = np.asarray(img).copy()
    os.remove(file_name)
    return torch.from_numpy(arr)
