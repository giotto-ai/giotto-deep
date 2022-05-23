from abc import ABC, abstractmethod
from typing import List, Any


import torch
from torch import nn


class SaveOutput(ABC):
    """ General class for saving outputs.
    Outputs will be stored in a list 'outputs'
    """

    @abstractmethod
    def __call__(self, module, module_in, module_out):
        pass

    def clear(self):
        self.outputs = []

    def get_outputs(self):
        return self.outputs


class SaveNodeOutput(SaveOutput):
    """ Class for saving activations of a node in
    a neural network.
    """
    outputs: List[Any]

    def __init__(self, entry:int=0):
        self.outputs = []
        self.entry = entry

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[:, self.entry].detach())


class SaveLayerOutput(SaveOutput):
    def __init__(self, entry:int=0):
        self.outputs = []

    def __call__(self, module: nn.Module, module_in, module_out):
        """ Add activation 

        Args:
            module ([type]):
                [description]
            module_in ([type]):
                [description]
            module_out ([type]):
                [description]
        """
        if isinstance(module_out, tuple):
            self.outputs.append(module_out[0].detach())
        else:
            self.outputs.append(module_out.detach())


