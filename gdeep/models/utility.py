import torch
from torch import nn


class SaveOutput:
    """ General class for saving outputs
    outputs will be stored in a list 'outputs'
    """
    def __init__(self):
        self.outputs = []

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
    def __init__(self, entry=0):
        super().__init__()
        self.entry = entry

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[:, self.entry].detach())


class SaveLayerOutput(SaveOutput):
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


class PeriodicNeuralNetwork(nn.Module):
    """Makes a periodic `nn.Module` of `nn`. `boundary_list` specifies
    the elementary region the periodic neural network is supported on.
    This creates neural networks
    factorized by the orbits of a group effect,
    which is given by reflection at the
    edges of the elementary domain.
    This class can be interpreted as an adapter for periodic neural networks.

    Args:
        nn (nn.Module):
            the standard torch module, your network
        boundary_list (list):
            list of pairs of floats, each
            defining the boundaries of a hypercube
    """

    def __init__(self, nn: nn.Module, boundary_list):
        super().__init__()
        self.nn = nn
        self.interval_length = torch.tensor([[b-a for a, b
                                              in boundary_list]])
        self.left_interval_bound = torch.tensor([[a for a, b
                                                  in boundary_list]])

    def forward(self, x_cont):
        x_cont = torch.abs(
                    torch.remainder(
                        x_cont-self.left_interval_bound,
                        2.*self.interval_length
                        ) - self.interval_length
                        ) + self.left_interval_bound
        x_cont = self.nn.forward(x_cont)
        return x_cont
