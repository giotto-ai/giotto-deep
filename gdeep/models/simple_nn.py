from typing import Callable, Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F

from gdeep.utility.custom_types import Tensor


# build a custom network class to easily do experiments
class FFNet(nn.Module):
    """Custom network class to easily do experiments

    It is a simple feed-forward network with a variable number of layers and
    neurons. The activation function is ReLU by default, but it can be changed
    by passing a different function as argument.
    The last layer is not activated.

    Args:
        arch (Tuple[int, ...], optional):
            The architecture of the network.
            Tuple containing the dimension of the layers
            inside your network. The default is (2, 3, 3, 2).
            The depth of the network is inferred from the length of the tuple.

        activation (Optional[Callable], optional):
            The activation function.
            Defaults to None. If None, ReLU is used.
            All the layers are activated with the same function. The last layer
            is not activated.
    """

    def __init__(
        self,
        arch: Tuple[int, ...] = (2, 3, 3, 2),
        activation: Optional[Callable] = None,
    ):
        super(FFNet, self).__init__()
        if activation is None:
            self.activation = F.relu
        else:
            self.activation = activation
        self.linears = nn.ModuleList(
            [nn.Linear(arch[i], arch[i + 1]) for i in range(len(arch) - 1)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, l in enumerate(self.linears):
            if i < len(self.linears) - 1:
                x = self.activation(l(x))
            else:
                x = l(x)
        return x
