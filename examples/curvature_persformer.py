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
from torch.optim.lr_scheduler import ExponentialLR

# Import the giotto-deep modules
from gdeep.data import CurvatureSamplingGenerator
from gdeep.topology_layers import SetTransformerOld, PersFormer, DeepSet, PytorchTransformer
from gdeep.topology_layers import AttentionPooling
from gdeep.pipeline import Pipeline
import json
# %%
# %%
curvatures = torch.tensor(np.load('data/curvatures_5000_1000_0_1.npy').astype(np.float32)).reshape(-1, 1)
diagrams = torch.tensor(np.load('data/diagrams_5000_1000_0_1.npy').astype(np.float32))[:, 999:, :2]

# %%

# dl_curvatures = DataLoader(TensorDataset(diagrams,
#                                          curvatures),
#                                          batch_size=32)
# %%
# class SmallDeepSet(nn.Module):
#     def __init__(self,
#         pool="sum",
#         dim_input=2,
#         dim_output=5,):
#         super().__init__()
#         self.enc = nn.Sequential(
#             nn.Linear(in_features=dim_input, out_features=16),
#             nn.ReLU(),
#             nn.Linear(in_features=16, out_features=32),
#             nn.ReLU(),
#             nn.Linear(in_features=32, out_features=64),
#             nn.ReLU(),
#             nn.Linear(in_features=64, out_features=16),
#         )
#         self.dec = nn.Sequential(
#             nn.Linear(in_features=16, out_features=32),
#             nn.ReLU(),
#             nn.Linear(in_features=32, out_features=64),
#             nn.ReLU(),
#             nn.Linear(in_features=64, out_features=32),
#             nn.ReLU(),
#             nn.Linear(in_features=32, out_features=dim_output),
#         )
#         self.ln = nn.LayerNorm(16)
#         self.pool = pool

#     def forward(self, x):
#         x = self.enc(x)
#         if self.pool == "max":
#             x = x.max(dim=1)[0]
#         elif self.pool == "mean":
#             x = x.mean(dim=1)
#         elif self.pool == "sum":
#             x = x.sum(dim=1)
#         x = self.dec(self.ln(x))
#         return x

# model = SmallDeepSet(dim_input=2, dim_output=1, pool="max")

model = SetTransformerOld(
    dim_input=2,
    num_outputs=1,  # for classification tasks this should be 1
    dim_output=1,  # number of classes
    dim_hidden=32,
    num_heads="4",
    layer_norm="False",  # use layer norm
    pre_layer_norm="False", # use pre-layer norm
    simplified_layer_norm="True",
    dropout_enc=0.0,
    dropout_dec=0.0,
    num_layer_enc=2,
    num_layer_dec=3,
    activation="gelu",
    bias_attention="True",
    attention_type="self_attention",
    layer_norm_pooling="False",
 )


# %%
# Do training and validation

# initialise loss
loss_fn = nn.MSELoss()

# Initialize the Tensorflow writer
writer = SummaryWriter(comment="Set Transformer curvature")

# initialise pipeline class
pipe = Pipeline(model, [dl_curvatures, None], loss_fn, writer)
# %%


# train the model
pipe.train(torch.optim.Adam,
           100,
           cross_validation=False,
           optimizers_param={"lr": 1e-3},
           lr_scheduler=ExponentialLR,
           scheduler_params={"gamma": 0.95})


# %%
x, y = next(iter(dl_curvatures))
x = x.to('cuda')
pred = pipe.model(x)

print(pred[-5:])
print(y[-5:])
# %%
model.eval()

for i in [3]:

    delta = torch.zeros_like(x[i].unsqueeze(0)).to('cuda')
    delta.requires_grad = True

    loss = pipe.model(x + delta).sum()
    loss.backward()

    import matplotlib.pyplot as plt

    c = torch.sqrt((delta.grad.detach().cpu()**2).sum(axis=-1))
    eps = 1
    c_max = c.max()


    sc = plt.scatter(x[i, :, 0].cpu(), x[i, :, 1].cpu(), c=-torch.log(c_max - c + eps))

    plt.colorbar(sc)
    plt.show()

# %%
def func(x):
    if x.shape[0] > 0:
        return np.max(x)
    else:
        return 0.0

for i in [0, 1, 3, 4]:
    x_life = x[i, :, 1] - x[i, :, 0]
    x_life = x_life.cpu().numpy()

    delta = torch.zeros_like(x[i].unsqueeze(0)).to('cuda')
    delta.requires_grad = True

    loss = pipe.model(x + delta).sum()
    loss.backward()

    c = torch.sqrt((delta.grad.detach().cpu()**2).sum(axis=-1))


    importance = c.squeeze().numpy()


    nbins = 10
    bins = np.linspace(0, x_life.max(), nbins+1)
    ind = np.digitize(x_life, bins)

    result = [func(importance[ind == j]) for j in range(1, nbins)]

    plt.plot(bins[:-2], result)
# %%
################################################################################################
#
#
#
#


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dl_curvatures, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print('[%d] loss: %.3f' %
            (epoch + 1, running_loss / i))

print('Finished Training')

# %%
x, y = next(iter(dl_curvatures))
x = x.to('cuda')
model(x)

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
import numba as nb
from math import tanh, cos, sin, sqrt, atanh, atan2

np_type = np.float64

@nb.jit
def cpu_dist_matrix(mat, curvature):
    m = mat.shape[0]

    out = np.empty((m, m), dtype=np_type)  # corrected dtype


    for i in range(m):
        out[i, i] = 0.0
        for j in range(i+1, m):
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
                out[j, i] = out[i, j]

    return out

# %%
x = np.random.rand(100, 2) * np.array([1.0, np.pi])
# %%

%timeit cpu_dist_matrix(x, -1.0)

# %%

%timeit pairwise_distances(x, metric=lambda x1, x2: geodesic_distance(-1.0, x1 , x2))
# %%

def remove_file(file_path):
    """Remove file in file_path if it exists"""
    try:
        os.remove(file_path)
    except OSError as ex:
        print("Error: {}, when removing {}.".format(ex.strerror, file_path))
# %%
curvature = 0.0
d1 = cpu_dist_matrix(x, curvature)
#np.fill_diagonal(d1, 0.0)
np.allclose(d1, pairwise_distances(x, metric=lambda x1, x2: geodesic_distance(curvature, x1 , x2)), atol=10, rtol=10.0)
# %%
import numpy as np
X = np.load('diagrams_5000_1000_0_1.npy')
# %%
import matplotlib.pyplot as plt

plt.scatter(X[0, :, 0], X[0, :, 1])
# %%
(X[0, :, 3] == 1.0).sum()
# %%
(X[4, :, 3] == 1.0).sum()
# %%
counter = 0
for i in range(X.shape[0]):
    if (X[i, :, -1] == 1.0).sum() != 301:
        #print("Warning")
        counter += 1
counter
# %%
y = np.load('curvatures_5000_1000_0_1.npy')
# %%
plt.hist(y)
# %%
! python -m pip install -U giotto-ph
# %%
import numpy as np
from gph import ripser_parallel

pc = np.random.random((10000, 2))
dgm = ripser_parallel(pc, maxdim=1, n_threads=12)
# %%
