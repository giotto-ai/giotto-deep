import numpy as np
import sys
import torch

if sys.version_info >= (3, 10):
    from typing import TypeAlias  # type: ignore
else:
    from typing_extensions import TypeAlias  # type: ignore

Tensor: TypeAlias = torch.Tensor
Array: TypeAlias = np.ndarray
