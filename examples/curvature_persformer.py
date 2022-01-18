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
from gdeep.topology_layers import Persformer, PersFormerOld, DeepSet, PytorchTransformer
from gdeep.topology_layers import AttentionPooling
from gdeep.pipeline import Pipeline
import json
# %%
# %%
curvatures = torch.tensor(np.load('data/curvatures_5000_1000_0_1.npy').astype(np.float32)).reshape(-1, 1)

# Only consider H1 features
diagrams = torch.tensor(np.load('data/diagrams_5000_1000_0_1.npy').astype(np.float32))[:, 999:, :2]

# %%


# %%

dl_curvatures_train = DataLoader(TensorDataset(diagrams[:4500],
                                         curvatures[:4500]),
                                         batch_size=32,
                                         shuffle=True)
dl_curvatures_val = DataLoader(TensorDataset(diagrams[4500:4750],
                                         curvatures[4500:4750]),
                                         batch_size=32)
dl_curvatures_test = DataLoader(TensorDataset(diagrams[-250:],
                                         curvatures[-250:]),
                                         batch_size=32)

model = Persformer(
    dim_input=2,
    num_outputs=1,  # for classification tasks this should be 1
    dim_output=1,  # number of classes
    dim_hidden=32,
    num_inds=32,
    num_heads="8",
    layer_norm="False",  # use layer norm
    pre_layer_norm="False", # use pre-layer norm
    simplified_layer_norm="False",
    dropout_enc=0.0,
    dropout_dec=0.0,
    num_layer_enc=2,
    num_layer_dec=4,
    activation="gelu",
    bias_attention="False",
    attention_type="self_attention",#"pytorch_self_attention_skip",
    layer_norm_pooling="False",
 )


# %%
# Do training and validation

# initialise loss
loss_fn = nn.MSELoss()

# Initialize the Tensorflow writer
writer = SummaryWriter(comment="Set Transformer curvature")

# initialise pipeline class
pipe = Pipeline(model, [dl_curvatures_train, dl_curvatures_val, dl_curvatures_test], loss_fn, writer)
# %%


# train the model
pipe.train(torch.optim.AdamW,
           50,
           cross_validation=False,
           optimizers_param={"lr": 1e-3, "weight_decay": 0.01},
           lr_scheduler=ExponentialLR,
           scheduler_params={"gamma": 0.9})

# %%
from sklearn.metrics import r2_score, mean_squared_error
model.eval()
x = torch.tensor(diagrams[-500:])
y = curvatures[-500:]
x = x.to('cuda')
pred = pipe.model(x)
pred = pred.clone().detach().cpu().numpy()

print('r2_score', r2_score(pred, y))
print('mse', mean_squared_error(pred, y))
# %%
from sklearn.metrics import r2_score, mean_squared_error
model.eval()
x = torch.tensor(diagrams[:500])
y = curvatures[:500]
x = x.to('cuda')
pred = pipe.model(x)
pred = pred.clone().detach().cpu().numpy()

mean_squared_error(pred, y)
# %%
import matplotlib.pyplot as plt

plt.scatter(y, pred)

# %%
from sklearn.metrics import r2_score, mean_squared_error
model.eval()
x = torch.tensor(diagrams[:500])
y = curvatures[:500]
x = x.to('cuda')
pred = pipe.model(x)
pred = pred.clone().detach().cpu().numpy()

mean_squared_error(pred, y)

# %%
# Check model performance on sample
# x, y = next(iter(dl_curvatures))
# x = x.to('cuda')
# pred = pipe.model(x)

# print(pred[-5:])
# print(y[-5:])

# %%
# Visualization of attention scores
from gdeep.models import ModelExtractor

me = ModelExtractor(pipe.model, loss_fn)

x = next(iter(dl_curvatures))[0]
list_activations = me.get_activations(x)
len(list_activations)

# %%
for activation in list_activations:
    print(activation.shape)

# %%
lista = me.get_layers_param()
for k, item in lista.items():
    print(k,item.shape)


# %%
from gdeep.analysis.interpretability import Interpreter
from gdeep.visualisation import Visualiser

inter = Interpreter(pipe.model, method="GuidedGradCam")
output = inter.interpret_image(next(iter(dl_curvatures))[0][0].reshape(1, 301, 2), 
                      0, pipe.model.enc[0].mab.fc_o);

# %%


# Saliency maps from scratch

x = next(iter(dl_curvatures_test))[0][0]


import matplotlib.pyplot as plt

c = torch.sqrt((output[0].cpu()**2).sum(axis=-1))



sc = plt.scatter(x[:, 0], x[:, 1], c=c)

plt.colorbar(sc)
plt.show()

# %%
pipe.model.eval()
x, y = next(iter(dl_curvatures_train))

x = x.to('cuda')

for i in [10, 12, 14, 15]:

    delta = torch.zeros_like(x[i].unsqueeze(0)).to('cuda')
    delta.requires_grad = True

    loss = pipe.model(x + delta).sum()
    loss.backward()

    import matplotlib.pyplot as plt

    c = torch.sqrt((delta.grad.detach().cpu()**2).sum(axis=-1))
    eps = 1
    c_max = c.max()


    #sc = plt.scatter(x[i, :, 0].cpu(), x[i, :, 1].cpu(), c=-torch.log(c_max - c + eps))
    sc = plt.scatter(x[i, :, 0].cpu(), x[i, :, 1].cpu(), c=c)

    plt.colorbar(sc)
    plt.savefig('plots/' + 'curvature' + str(y[i].item()) + '.pdf')
    plt.show()
    print('y:', y[i], ', pred:', pipe.model(x)[i])

# %%
def func(x):
    if x.shape[0] > 0:
        return np.max(x)
    else:
        return 0.0

result = [0] * 11

for i in range(0, 32):
    x_life = x[i, :, 1] - x[i, :, 0]
    x_life = x_life.cpu().numpy()
    x_life_norm = x_life / x_life.max()

    delta = torch.zeros_like(x[i].unsqueeze(0)).to('cuda')
    delta.requires_grad = True

    loss = pipe.model(x + delta).sum()
    loss.backward()

    c = torch.sqrt((delta.grad.detach().cpu()**2).sum(axis=-1))


    importance = c.squeeze().numpy()
    importance_norm = importance / importance.max()


    nbins = 10+1
    bins = np.linspace(0, x_life_norm.max(), nbins+1)
    ind = np.digitize(x_life_norm, bins)

    result = [func(importance_norm[ind == j]) / 32 + result[j] for j in range(0, nbins)]

plt.plot([i / (nbins - 1) for i in list(range(0, nbins-1))], result[1:])

plt.xlabel("relative lifetime")
plt.ylabel("relative importance score")
plt.savefig('plots/' + 'lifetime_importance_max.pdf')
#plt.show()

# %%
plt.savefig('plots/' + 'lifetime_importance_sum.pdf')
# %%
