import torch.nn as nn
import torch.nn.functional as F


# build a custom network class to easily do experiments
class FFNet(nn.Module):
    """This is the class to build a custom feed
    forward network that is easily built from an array,
    in which the number and dimensions of the layers is specified.

    Args:
        verbose (int): bool, default = 0;
            set this to 1 for debugging
        arch (list): list of int or 1d array, default=[2,3,3];
            this is the list containing the dimension of the layers
            inside your network. all laysers have ``relu`` except for
            the last one which has ``sigmoid`` as activation function.
            The first number is the dimension of the input! Thhe final
            of the output
    """

    def __init__(self, verbose=0, arch=[2, 3, 3, 2]):
        super(FFNet, self).__init__()
        self.verbose = verbose
        self.arch = arch
        for i, in_f in enumerate(arch):
            if i < len(arch) - 1:
                val = "self.layer" + str(i) + "=" + \
                   "nn.Linear("+str(in_f) +","+str(arch[i+1])+")"
                exec(val)
                val2 = "self.layer"+str(i)+".weight.data.uniform_(-1, 1)"
                eval(val2)
                val3 = "self.layer"+str(i)+".bias.data.uniform_(-1, 1)"
                eval(val3)

    def forward(self, x_cont):
        # output_vars = []
        for i, in_f in enumerate(self.arch):
            if i < len(self.arch) - 1:
                if i == 0:
                    val = "x" + str(i) + "=F.relu(" + \
                        "self.layer" + str(i) + "(x_cont)" + ")"
                # put softmax in last layer
                elif i == len(self.arch) - 2:
                    val = "x"+str(i)+"=F.softmax(" + \
                        "self.layer"+str(i)+"(x" + str(i - 1) + \
                            ")" + ",dim=-1)"
                else:
                    val = "x" + str(i) + "=F.relu(" + \
                        "self.layer" + str(i) + "(x" + \
                            str(i - 1) + ")" + ")"
                if self.verbose:
                    print(val)
                try:
                    exec(val)
                except:
                    raise ValueError("The dimensions of x" + str(i)
                                     + " are not correct!")
                if self.verbose:
                    print("x" + str(i) + " size: ",
                          eval("x" + str(i) + ".shape"))
        i -= 1
        return eval("x" + str(i))
