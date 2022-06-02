from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class PeriodicNeuralNetwork(nn.Module):
    """Makes a periodic `nn.Module` of `nn`. `boundary_list` specifies
    the elementary region the periodic neural network is supported on.
    This creates neural networks
    factorized by the orbits of a group effect,
    which is given by reflection at the
    edges of the elementary domain.
    This class can be interpreted as an adapter for periodic neural networks.

    Args:
        nn :
            the standard torch module, your network
        boundary_list :
            list of pairs of floats, each
            defining the boundaries of a hypercube
    """

    def __init__(self, nn: nn.Module, boundary_list: List[Tuple[float, float]]) -> None:
        super().__init__()
        self.nn = nn
        self.interval_length = torch.tensor([[b - a for a, b in boundary_list]])
        self.left_interval_bound = torch.tensor([[a for a, b in boundary_list]])

    def forward(self, x_cont: Tensor) -> Tensor:
        x_cont = (
            torch.abs(
                torch.remainder(
                    x_cont - self.left_interval_bound, 2.0 * self.interval_length
                )
                - self.interval_length
            )
            + self.left_interval_bound
        )
        x_cont = self.nn.forward(x_cont)
        return x_cont
