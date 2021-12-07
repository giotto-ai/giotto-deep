# %%
from IPython import get_ipython  # type: ignore

# %% 
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# %%

from dotmap import DotMap
import json

# Import the PyTorch modules
import torch  # type: ignore
from torch import nn  # type: ignore
from torch.optim import SGD, Adam, RMSprop  # type: ignore

import numpy as np # type: ignore

# Import Tensorflow writer
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader, TensorDataset

# Import the giotto-deep modules
from gdeep.data import CurvatureSamplingGenerator
from gdeep.topology_layers import SetTransformer, PersFormer, DeepSet, PytorchTransformer
from gdeep.topology_layers import AttentionPooling
from gdeep.pipeline import Pipeline
import json
# %%
num_points = 10
cg = CurvatureSamplingGenerator(num_samplings=num_points,
                        num_points_per_sampling=200,
                        homology_dimensions=(1,))
curvatures = cg.get_curvatures()
diagrams = cg.get_diagrams()
#np.save('curvatures_10k.npy', curvatures)
#np.save('diagrams_curvature_10k.npy', diagrams)
# %%
#curvatures = torch.tensor(np.load('curvatures.npy').astype(np.float32)).reshape(-1, 1)
#diagrams = torch.tensor(np.load('diagrams_curvature.npy').astype(np.float32))

dl_curvatures = DataLoader(TensorDataset(torch.tensor(diagrams[:, :, :2], dtype=torch.float32),
                                         torch.tensor(curvatures, dtype=torch.float32)),
                                         batch_size=10)
# %%
class SmallDeepSet(nn.Module):
    def __init__(self,
        pool="sum",
        dim_input=2,
        dim_output=5,):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=dim_input, out_features=16),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=dim_output),
        )
        self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        x = self.dec(x)
        return x

model = SmallDeepSet(dim_input=2, dim_output=1, pool="sum")

# %%
# Do training and validation

# initialise loss
#loss_fn = nn.MSELoss()

# Initialize the Tensorflow writer
#writer = SummaryWriter(comment=json.dumps(config_model.toDict())\
#                                + json.dumps(config_data.toDict()))
#writer = SummaryWriter(comment="deep set")

# initialise pipeline class
#pipe = Pipeline(model, [dl_curvatures, None], loss_fn, writer)
# %%


# train the model
# pipe.train(torch.optim.Adam,
#            1000,
#            cross_validation=False,
#            optimizers_param={"lr": 1e-3})
# %%

model = nn.Sequential(
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )

model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100000):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dl_curvatures, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = su.detach(), labels.to('cuda').reshape(num_points, 1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss))

print('Finished Training')
# %%
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

for epoch in range(100000):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dl_curvatures, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = su.detach(), labels.to('cuda').reshape(num_points, 1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss))

print('Finished Training')


# %%
model = nn.Sequential(
            nn.Linear(in_features=10, out_features=1),
        )
model.cuda()
num_points = 10
data = (torch.rand(num_points, 10).cuda(), torch.rand(num_points).reshape(num_points, 1))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(1000):  # loop over the dataset multiple times

    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    inputs, labels = inputs.to('cuda'), labels.to('cuda'

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss = loss.item()
    print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss))

print('Finished Training')

# %%
import matplotlib.pyplot as plt

plt.scatter(diagrams[0, :, 0], diagrams[0, :, 1])
# %%

import numpy as np
import numba

def geodesic_distance(curvature, x1 , x2):
    
    if curvature > 0:
        R = 1/np.sqrt(curvature)
        v1 = np.array([R * np.sin(x1[0]/R) * np.cos(x1[1]), 
                    R * np.sin(x1[0]/R) * np.sin(x1[1]),
                    R * np.cos(x1[0]/R)])
        
        v2 = np.array([R * np.sin(x2[0]/R) * np.cos(x2[1]), 
                    R * np.sin(x2[0]/R) * np.sin(x2[1]),
                    R * np.cos(x2[0]/R)])

        
        dist = R * np.arctan2(np.linalg.norm(np.cross(v1,v2)), (v1*v2).sum())
    
    elif curvature == 0:
        v1 = np.array([x1[0]*np.cos(x1[1]), x1[0]*np.sin(x1[1])])
        v2 = np.array([x2[0]*np.cos(x2[1]), x2[0]*np.sin(x2[1])])
        dist = np.linalg.norm( (v1 - v2) )  
    
    elif curvature < 0:
        R = 1/np.sqrt(-curvature)
        z = np.array([ np.tanh(x1[0]/(2 * R)) * np.cos(x1[1]),
                    np.tanh(x1[0]/(2 * R)) * np.sin(x1[1])])
        w = np.array([np.tanh(x2[0]/(2 * R)) * np.cos(x2[1]),
                    np.tanh(x2[0]/(2 * R)) * np.sin(x2[1])])
        temp = np.linalg.norm([(z*w).sum() - 1, np.linalg.det([z,w]) + 1])
        dist = 2 * R * np.arctanh(np.linalg.norm(z - w)/temp)
        
    return dist
# %%
x = np.random.rand(1_000, 2)
%timeit geodesic_distance(-1, x, x)
# %%
%timeit geodesic_distance(0, x1 , x2)
# %%
%timeit np.linalg.norm( x1 - x2 ) 
# %%
from sklearn.metrics import pairwise_distances
x = np.random.rand(10, 2)
%timeit pairwise_distances(x, metric=lambda x1, x2: geodesic_distance(-1, x1 , x2))
# %%
@numba.jit(nopython=True)
def euclidean_numba1(x):
    """Euclidean square distance matrix using pure loops
    and no NumPy operations
    """
    num_samples, num_feat = x.shape
    dist_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            v1_x, v1_y = x[i, 0]*np.cos(x[i, 1]), x[i, 0]*np.sin(x[i, 1])
            v2_x, v2_y = x[j, 0]*np.cos(x[j, 1]), x[j, 0]*np.sin(x[j, 1])
            dist = np.sqrt((v1_x-v2_x)**2 + (v1_y-v2_y)**2)  
            dist_matrix[i][j] = dist
    return dist_matrix
# %%
%timeit euclidean_numba1(x)
# 19.4 ms ± 38.4 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

# %%
import numpy as np
from numba import cuda
from math import tanh, cos, sin, sqrt, atanh, atan2

USE_64 = True

if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32
    

@cuda.jit("void(float{}[:, :], float{}[:, :], float{})".format(bits, bits, bits))
def distance_matrix(mat, out, curvature):
    m = mat.shape[0]
    i, j = cuda.grid(2)
    if i < m and j < m:
        if curvature > 0:
            R = 1.0/sqrt(curvature)
            z0 = R * sin(mat[i, 0] / R) * cos(mat[i, 1])
            z1 = R * sin(mat[i, 0] / R) * sin(mat[i, 1])
            z2 = R * cos(mat[i, 0] / R)
            
            w0 = R * sin(mat[j, 0] / R) * cos(mat[j, 1])
            w1 = R * sin(mat[j, 0] / R) * sin(mat[j, 1])
            w2 = R * cos(mat[j, 0] / R)
            
            cross0 = z1 * w2 - z2 * w1
            cross1 = z2 * w0 - z0 * w2
            cross2 = z0 * w1 - z1 * w0
            
            out[i, j] = R * atan2(sqrt(cross0 * cross0 + cross1 * cross1 + cross2 * cross2),
                                  z0 * w0 + z1 * w1 + z2 * w2)
        
        if curvature < 0:
            R = 1.0/sqrt(-curvature)
            z0 = tanh(mat[i, 0]/(2.0 * R)) * cos(mat[i, 1])
            z1 = tanh(mat[i, 0]/(2.0 * R)) * sin(mat[i, 1])
            w0 = tanh(mat[j, 0]/(2.0 * R)) * cos(mat[j, 1])
            w1 = tanh(mat[j, 0]/(2.0 * R)) * sin(mat[j, 1])
            
            temp0 = z0 * w0 + z1 * w1 - 1.0
            temp1 = z0 * w1 - z1 * w0 + 1.0
            temp = sqrt(temp0 * temp0 + temp1 * temp1)
            x = sqrt((z0 - w0) * (z0 - w0) + (z1 - w1) * (z1 - w1))/temp
            out[i, j] = 2.0 * R * atanh(x)
            
        if curvature == 0.0:  # it does not make sense to compare floats
            z0 = mat[i, 0] * cos(mat[i, 1])
            z1 = mat[i, 0] * sin(mat[i, 1])
            
            w0 = mat[j, 0] * cos(mat[j, 1])
            w1 = mat[j, 0] * sin(mat[j, 1])
            
            out[i, j] = sqrt((z0 - w0) * (z0 - w0) + (z1 - w1) * (z1 - w1))

def gpu_dist_matrix(mat, curvature):
    rows = mat.shape[0]

    block_dim = (16, 16)
    grid_dim = (int(rows/block_dim[0] + 1), int(rows/block_dim[1] + 1))

    stream = cuda.stream()
    mat2 = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
    out2 = cuda.device_array((rows, rows))
    distance_matrix[grid_dim, block_dim](mat2, out2, curvature)
    out = out2.copy_to_host(stream=stream)

    return out
# %%
x = np.random.rand(1000, 2) * np.array([1.0, np.pi])
# %%

gpu_dist_matrix(x, 1.0).shape

# %%
pairwise_distances(x, metric=lambda x1, x2: geodesic_distance(-1.0, x1 , x2)).shape
# %%

def remove_file(file_path):
    """Remove file in file_path if it exists"""
    try:
        os.remove(file_path)
    except OSError as ex:
        print("Error: {}, when removing {}.".format(ex.strerror, file_path))
