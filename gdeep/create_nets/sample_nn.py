import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """This functions creates a simple neural network with
    three fully connected layers of dimensions 2 - 10 - 2.
        
    """
    
    def __init__(self, nodes_layer_1 = 8, dropout_p=0.0):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, nodes_layer_1, bias=True)
        self.drop_layer = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(nodes_layer_1, 1, bias=True)

        # Initialize weights to zero
        self.fc1.weight.data.fill_(0.)
        self.fc2.weight.data.fill_(0.)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.drop_layer(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class DeeperNN(nn.Module):
    """This functions creates a deeper neural network with
        four fully connected layers of dimensions 2 - 16 - 32
        -64 - 2.
        
        """
    
    def __init__(self, dropout_p=0.0):
        super(DeeperNN, self).__init__()
        self.fc1 = nn.Linear(2, 16, bias=True)
        self.drop_layer = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(16, 32, bias=True)
        self.drop_layer = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(32, 64, bias=True)
        self.drop_layer = nn.Dropout(p=dropout_p)
        self.fc4 = nn.Linear(64, 2, bias=True)

        # Initialize weights to zero
        self.fc1.weight.data.fill_(0.)
        self.fc2.weight.data.fill_(0.)
        self.fc3.weight.data.fill_(0.)
        self.fc4.weight.data.fill_(0.)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.leaky_relu(self.fc2(x))
        return x


class LogisticRegressionNN(nn.Module):
    """This functions creates a logistic regression neural network
    """
    
    def __init__(self, dim_input=2):
        super(LogisticRegressionNN, self).__init__()
        self.fc1 = nn.Linear(dim_input, 1, bias=True)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return x


# build a custom network class to easily do experiments
class Net(nn.Module):
    '''This is the custom network that is easily built from an array,
    in which the number and dimensions of the layers is specified.
    (from Matteo)
    '''
    def __init__(self,verbose = 0, arch=[2,3,3]):
        '''
        Parameters
        ----------
        
         - verbose: bool, default = 0;
             set this to 1 for debugging
         - arch: list of int or 1d array, default=[2,3,3];
             this is the list containing the dimension of the layers
             inside your network. all laysers have ``relu`` except for
             the last one which has ``sigmoid`` as activation function.
             The fsrt number is the dimension of the input! No need to
             specify the output dimension of 1
        '''
        super(Net, self).__init__()
        self.verbose = verbose
        self.arch = arch
        for i,in_f in enumerate(arch):
            try:
                val = "self.layer"+str(i)+"="+\
                "nn.Linear("+str(in_f) +","+str(arch[i+1])+")"
                exec(val)
                val2 = "self.layer"+str(i)+".weight.data.uniform_(-1, 1)"
                eval(val2)
                val3 = "self.layer"+str(i)+".bias.data.uniform_(-1, 1)"
                eval(val3)
            except:
                val = "self.layer"+str(i)+"="+\
                "nn.Linear("+str(in_f) +",2)"
                exec(val)
                val2 = "self.layer"+str(i)+".weight.data.uniform_(-1, 1)"
                eval(val2)
                val3 = "self.layer"+str(i)+".bias.data.uniform_(-1, 1)"
                eval(val3)

    def forward(self, x_cat, x_cont):
        #output_vars = []
        for i,in_f in enumerate(self.arch):
            if i == 0:
                val = "x"+str(i)+"=F.relu("+\
                "self.layer"+str(i)+"(x_cont)"+")"
            # put sigmoid in last layer
            elif i == len(self.arch)-1:
                val = "x"+str(i)+"=F.softmax("+\
                "self.layer"+str(i)+"(x"+str(i-1)+")"+",dim=-1)"
            else:
                val = "x"+str(i)+"=F.relu("+\
                "self.layer"+str(i)+"(x"+str(i-1)+")"+")"
            if self.verbose:
                print(val)
            try:
                exec(val)
            except:
                raise ValueError("The dimensions of x"+str(i)+" are not correct!")
            if self.verbose:
                print("x"+str(i)+" size: ",eval("x"+str(i)+".shape"))
        return eval("x"+str(i))


class CircleNN(nn.Module):
    """ Generates a neural network that is proportional to a Gaussian Kernel in 2D
    """
    def __init__(self):
        super().__init__()
        
        self.dim = 2

                
    def forward(self, x_cont):
        try:
            assert(x_cont.shape[-1]==2)
        except:
            raise ValueError(f'input has to be a {2}-dimensional vector')
        activation = 0.5*torch.exp(-torch.sum(x_cont**2, axis=-1)+1)-0.5
        return activation.reshape((-1,1))
    
    def return_input_dim(self):
        return 2


class CircleNN3D(nn.Module):
    """ Generates a neural network that is proportional to a Gaussian Kernel in 3D
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_cont):
        activation = torch.exp(x_cont**2)
        return activation


class SampleCNN_MNIST_SAMPLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SampleCNN_MNIST_SAMPLE_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# class ListModule(nn.Module):
#     """
#     cf https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
#     Args:
#         nn ([type]): [description]
#     """
#     def __init__(self, *args):
#         super(ListModule, self).__init__()
#         idx = 0
#         for module in args:
#             self.add_module(str(idx), module)
#             idx += 1

#     def __getitem__(self, idx):
#         if idx < 0 or idx >= len(self._modules):
#             raise IndexError('index {} is out of range'.format(idx))
#         it = iter(self._modules.values())
#         for i in range(idx):
#             next(it)
#         return next(it)

#     def __iter__(self):
#         return iter(self._modules.values())

#     def __len__(self):
#         return len(self._modules)


# class ConstructorNN(ListModule):
#     """ This Constructor creates a fully connected 
#     neural network for binary classification from
#     an array of the width of every layer
#     """
#     def __init__(self, layer_widths, verbose=True):
#         try:
#             assert(len(layer_widths>0))
#         except:
#             print("The layer_widths is not a valid input")

#         layer_widths.append(1)

#         super(ConstructorNN, self).__init__()
#         layers = []

#         for layer_number, layer_width in enumerate(layer_widths[1:]):
#             layers.append(nn.Linear(layer_widths[layer_number],layer_width))


#             if verbose:
#                 print("Appended nn.Linear", (layer_widths[layer_number],layer_width))

#         self.layers = ListModule(*layers)

    
#     def forward(self, x):
#         for layer in self.layers[:-1]:
#             x = F.relu(layer(x))

#         return F.sigmoid(self.layers[-1](x))
