import torch.nn as nn
import torch.nn.functional as F


# build a custom network class to easily do experiments
class FFNet(nn.Module):
    """This is the class to build a custom feed
    forward network that is easily built from an array,
    in which the number and dimensions of the layers is specified.

    Args:
        arch (list of int or 1d array, default=[2,3,3,2]):
            this is the list containing the dimension of the layers
            inside your network. all laysers have ``relu`` except for
            the last one which has ``sigmoid`` as activation function.
            The first number is the dimension of the input! Thhe final
            of the output
    """

    def __init__(self, arch=(2, 3, 3, 2)):
        super(FFNet, self).__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(arch[i], arch[i + 1]) for i in range(len(arch) - 1)]
        )

    def forward(self, x):
        for i, l in enumerate(self.linears[:-1]):
            x = F.relu(l(x))
        x = self.linears[-1](x)
        return x
