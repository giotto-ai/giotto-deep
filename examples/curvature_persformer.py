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
cg = CurvatureSamplingGenerator(num_samplings=24,
                        num_points_per_sampling=100)
curvatures = cg.get_curvatures()
diagrams = cg.get_diagrams()
np.save('curvatures.npy', curvatures)
np.save('diagrams_curvature.npy', diagrams)
# %%
curvatures = torch.tensor(np.load('curvatures.npy').astype(np.float32)).reshape(-1, 1)
diagrams = torch.tensor(np.load('diagrams_curvature.npy').astype(np.float32))

dl_curvatures = DataLoader(TensorDataset(diagrams, curvatures),
                   batch_size=2)
# %%
class SmallDeepSet(nn.Module):
    def __init__(self,
        pool="max",
        dim_input=2,
        dim_output=5,):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=dim_input, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=dim_output),
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

model = SmallDeepSet(dim_input=4, dim_output=1)
# %%
# Do training and validation

# initialise loss
loss_fn = nn.L1Loss()

# Initialize the Tensorflow writer
#writer = SummaryWriter(comment=json.dumps(config_model.toDict())\
#                                + json.dumps(config_data.toDict()))
writer = SummaryWriter(comment="deep set")

# initialise pipeline class
pipe = Pipeline(model, [dl_curvatures, None], loss_fn, writer)
# %%


# train the model
pipe.train(torch.optim.Adam,
           10,
           cross_validation=False,
           optimizers_param={"lr": 5e-4})
# %%
import matplotlib.pyplot as plt

plt.scatter(diagrams[1, :, 0], diagrams[1, :, 1])
# %%
