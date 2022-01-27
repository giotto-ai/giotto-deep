
from PIL import Image
import numpy as np
import os
# from plotly.io import write_image
import io
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
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    arr = np.asarray(img)
    return torch.from_numpy(arr)


def png2tensor(file_name) -> torch.Tensor:
    img = Image.open(file_name)
    arr = np.asarray(img).copy()
    os.remove(file_name)
    return torch.from_numpy(arr)
