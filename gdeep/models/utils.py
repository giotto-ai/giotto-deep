from abc import ABC, abstractmethod
from typing import List, Any, Union

import torch
from torch import nn

from gdeep.utility.custom_types import Tensor


class SaveOutput(ABC):
    """General class for saving outputs of
    neural networks.
    Outputs will be stored in a list 'outputs'
    """

    outputs: List[Any]

    @abstractmethod
    def __call__(
        self,
        module: nn.Module,
        module_in: Tensor,
        module_out: Tensor,
    ) -> None:
        pass

    def clear(self) -> None:
        self.outputs = []

    def get_outputs(self) -> List[Any]:
        return self.outputs


class SaveNodeOutput(SaveOutput):
    """Class for saving activations of a node in
    a neural network.

    Args:
        entry:
            select the entry row in the output
            of the layer, i.e. the node output
    """

    def __init__(self, entry: int = 0):
        self.outputs = []
        self.entry = entry

    def __call__(self, module: nn.Module, module_in: Tensor, module_out: Tensor):
        self.outputs.append(module_out[:, self.entry].detach())


class SaveLayerOutput(SaveOutput):
    def __init__(self) -> None:
        self.outputs = []

    def __call__(
        self,
        module: nn.Module,
        module_in: Tensor,
        module_out: Tensor,
    ) -> None:
        """This function stores the layer activations

        Args:
            module :
                the neural network model (nn.Module)
            module_in:
                the input tensor of the module
            module_out:
                the output tensor of the module
        """
        if isinstance(module_out, tuple):
            if hasattr(module_out[0], "detach"):
                self.outputs.append(module_out[0].detach())
            else:
                self.outputs.append(module_out[0])  # .detach())
        else:
            if hasattr(module_out, "detach"):
                self.outputs.append(module_out.detach())
            else:
                self.outputs.append(module_out)  # .detach())
