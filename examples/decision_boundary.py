#%%
# Create binary data cloud


import math

seed=42

import numpy as np
from sklearn import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F


from gtda.plotting import plot_point_cloud

import matplotlib.pyplot as plt

#from gdeep.heatmaps import *
#from scipy.spatial.distance import KDTree
# %%
data, label = datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=seed)

A = data[label==0]
B = data[label==1] #+ [2,0]
# %%
plot_point_cloud(np.concatenate((A, B)))
# %%
n_samples = 1000
sample_points = np.random.rand(n_samples, 2)

sample_points = sample_points.dot(np.diag([2,2])) + np.array([-1,-1])
# %%
plot_point_cloud(np.concatenate((A, B,sample_points)))
# %%
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1])
plt.scatter(sample_points[:,0],sample_points[:,1])

plt.show()
# %%
# Train a simple logistic regression model for the binary classification task

class LogisticRegressionNN(nn.Module):
    """This functions creates a logistic regression neural network
    """
    
    def __init__(self, dim_input=2):
        super(LogisticRegressionNN, self).__init__()
        self.fc1 = nn.Linear(dim_input, 1, bias=True)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return x


class ListModule(nn.Module):
    """
    cf https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    Args:
        nn ([type]): [description]
    """
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class ConstructorNN(ListModule):
    """ This Constructor creates a fully connected 
    neural network for binary classification from
    an array of the width of every layer
    """
    def __init__(self, layer_widths, verbose=True):
        try:
            assert(len(layer_widths>0))
        except:
            print("The layer_widths is not a valid input")

        layer_widths.append(1)

        super(ConstructorNN, self).__init__()
        layers = []

        for layer_number, layer_width in enumerate(layer_widths[1:]):
            layers.append(nn.Linear(layer_widths[layer_number],layer_width))


            if verbose:
                print("Appended nn.Linear", (layer_widths[layer_number],layer_width))

        self.layers = ListModule(*layers)

    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        return F.sigmoid(self.layers[-1](x))



# build a custom network class to easily do experiments
class Net(nn.Module):
    '''This is the custom network that is easily built from an array,
    in which the number and dimensions of the layers is specified.
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
                "nn.Linear("+str(in_f) +",1)"
                exec(val)
                val2 = "self.layer"+str(i)+".weight.data.uniform_(-1, 1)"
                eval(val2)
                val3 = "self.layer"+str(i)+".bias.data.uniform_(-1, 1)"
                eval(val3)

    def forward(self, x_cont):
        output_vars = []
        for i,in_f in enumerate(self.arch):
            if i == 0:
                val = "x"+str(i)+"=F.relu("+\
                "self.layer"+str(i)+"(x_cont)"+")"
            # put sigmoid in last layer
            elif i == len(self.arch)-1:
                val = "x"+str(i)+"=torch.sigmoid("+\
                "self.layer"+str(i)+"(x"+str(i-1)+")"+")"
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





# %%

log_reg = LogisticRegressionNN(2)

circle_detect = Net(0, [2,10,10])

X_train = torch.from_numpy(np.concatenate((A, B))).float()
y_train = torch.from_numpy(np.concatenate((np.ones(A.shape[0]),\
                np.zeros(B.shape[0])))).float()

print(log_reg)
print(circle_detect)

train_classification_nn(circle_detect, X_train, y_train, n_epochs=20000)
# %%

epsilon = 0.01
n_epochs = 1000

sample_points_tensor = torch.from_numpy(sample_points).float()

for _ in range(0,n_epochs):

    delta = torch.zeros_like(sample_points_tensor, requires_grad=True)

    predict = circle_detect.forward(sample_points_tensor + delta)

    loss = torch.sum((predict-0.5)**2)

    loss.backward()

    sample_points_tensor -= epsilon * delta.grad.detach()

delta = torch.zeros_like(sample_points_tensor, requires_grad=True)

predict = circle_detect.forward(sample_points_tensor+delta)
loss = torch.sum((predict-0.5)**2)

loss.backward()


sample_points_tensor = sample_points_tensor[\
    torch.stack((
    #torch.norm(delta.grad.detach(), dim=1)> 1.e-05,\
    (1.-predict>1e-2)[:,0],\
    (predict>1e-2)[:,0]\
    ),dim=1).all(dim=1)\
    ]

sample_points_new = sample_points_tensor.numpy()


# plt.scatter(A[:,0],A[:,1])
# plt.scatter(B[:,0],B[:,1])
# plt.scatter(sample_points[:,0],sample_points[:,1])
# plt.show()

plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1])
#plt.scatter(sample_points[:,0],sample_points[:,1])
plt.scatter(sample_points_new[:,0],sample_points_new[:,1])

plt.show()
# %%
# Plot log_reg

delta = 0.1
x = np.arange(-1.5, 1.5, delta)
y = np.arange(-1.5, 1.5, delta)
X, Y = np.meshgrid(x, y)

X_tensor, Y_tensor = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
X_tensor = X_tensor.reshape((X_tensor.shape[0],X_tensor.shape[1],1))
Y_tensor = Y_tensor.reshape((X_tensor.shape[0],X_tensor.shape[1],1))
XY_tensor = torch.cat((X_tensor,Y_tensor), 2)

XY_tensor = XY_tensor.reshape((-1,2))

Z_tensor = circle_detect.forward(XY_tensor)#nn.Softmax(dim=0)(XY_tensor)[:,0]
Z_tensor = Z_tensor.reshape((X_tensor.shape[0],X_tensor.shape[1]))
#torch.div(torch.exp(XY_tensor)[:,:,0],torch.sum(torch.exp(XY_tensor), axis=2))
Z = Z_tensor.detach().numpy()
plt.contourf(X, Y, Z)

# %%
def make_torus_point_cloud(label: int, n_points: int, noise: float,\
    rotation: Rotation, base_point: np.array, radius: float=1.):
    """Generate point cloud of a torus

    Args:
        label (int): label of the data points
        n_points (int): number of sample points for each direction
        noise (float): noise
        rotation: Rotation
        base_point (np.array): center of the torus
        radius: float

    Returns:
        (np.array, np.array): data_points, labels
    """
    torus_point_clouds = np.asarray(
            [
                [
                    (2 + np.cos(s)) * np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                    (2 + np.cos(s)) * np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.sin(s) + noise * (np.random.rand(1)[0] - 0.5),
                ]
                for t in range(n_points)
                for s in range(n_points)
            ]
        )
    
    # print(
    # torus_point_clouds.shape,
    # rotation.rotation_matrix().shape
    # )
    torus_point_clouds = np.einsum("ij,kj->ki",  rotation.rotation_matrix(), torus_point_clouds)

    torus_point_clouds += base_point

    # label tori with 2
    torus_labels = label * np.ones(n_points)

    return torus_point_clouds, torus_labels



torus_point_clouds, torus_labels = make_torus_point_cloud(0, 20, 0.0,\
    Rotation(1,2,math.pi/2), np.array([[1,0,0]]))
torus_point_clouds_2, torus_labels_2 = make_torus_point_cloud(1, 20, 0.0,\
    Rotation(1,2,0), np.array([[2,0,0]]))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

data_points = np.concatenate((torus_point_clouds,torus_point_clouds_2), axis=0)
labels = np.concatenate((torus_labels,torus_labels_2), axis=0)
ms = ['o' if lab==1. else '^' for lab in list(labels)]

#ax.scatter(data_points[:,0], data_points[:,1], data_points[:,2], marker=ms)
ax.scatter(torus_point_clouds[:,0], torus_point_clouds[:,1], torus_point_clouds[:,2])
ax.scatter(torus_point_clouds_2[:,0], torus_point_clouds_2[:,1], torus_point_clouds_2[:,2])

plt.show()

# plot_point_cloud(data_points,\
#     np.concatenate((torus_labels, torus_labels_2), axis=0))
# %%

class Rotation():
    def __init__(self, axis_0, axis_1, angle):
        self._axis_0 = axis_0
        self._axis_1 = axis_1
        self._angle = angle

    def return_axis(self, idx):
        return eval('self._axis_'+str(idx))
    
    def return_angle(self):
        return self._angle

    def rotation_matrix(self):
        rotation_matrix = np.identity(3)
        rotation_matrix[self._axis_0,self._axis_0]\
            = math.cos(self._angle)
        rotation_matrix[self._axis_1,self._axis_1]\
            = math.cos(self._angle)
        rotation_matrix[self._axis_1,self._axis_0]\
            = math.sin(self._angle)
        rotation_matrix[self._axis_0,self._axis_1]\
            = -math.sin(self._angle)
        return rotation_matrix
# %%
