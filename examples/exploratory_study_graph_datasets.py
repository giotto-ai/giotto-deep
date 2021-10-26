# %%
import time
from os.path import join
import numpy as np
import torch
import matplotlib.pyplot as plt  # type: ignore
from sklearn.manifold import MDS  # type: ignore
from matplotlib.pylab import matshow  # type: ignore
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore

from gtda.diagrams import PairwiseDistance  # type: ignore
from gdeep.topology_layers import load_data

from sklearn.metrics import pairwise_distances # type: ignore

# %%
dataset_name = "MUTAG"

x_pds_dict, x_features, y = load_data(dataset_name)


# %%
# transform x_pds to a single tensor with tailing zeros
num_types = x_pds_dict[0].shape[1] - 2
num_graphs = len(x_pds_dict.keys())  # type: ignore

max_number_of_points = max([x_pd.shape[0]
                            for _, x_pd in x_pds_dict.items()])  # type: ignore

x_pds = torch.zeros((num_graphs, max_number_of_points, num_types + 2))

for idx, x_pd in x_pds_dict.items():  # type: ignore
    x_pds[idx, :x_pd.shape[0], :] = x_pd
# %%
# compute pairwise bottleneck distance of diagrams
# takes about 200 seconds = 3:20 minutes for COX2
diagrams = []
t = time.time()
n_graphs = len(x_pds_dict.keys())  # type: ignore
for i in range(n_graphs):
    x = torch.max(x_pds_dict[i][:, 2:], dim=-1).indices
    x_pts = x_pds_dict[i][:, :2]
    diagrams.append(torch.cat((x_pts, x.reshape(-1, 1)), dim=-1)
                    .detach().numpy())

distances = np.zeros(())
for type_ in range(int(max(diagrams[0][:, 2]) + 1.0)):
    max_size = max([(diagrams[idx][:, 2] == type_).sum()
                    for idx in range(len(diagrams))])
    pds = []
    for idx in range(len(diagrams)):
        pd_type = diagrams[idx][diagrams[idx][:, 2] == type_]
        pd_type_pad = np.pad(pd_type,
                             ((0, max_size - pd_type.shape[0]), (0, 0)),
                             'constant')
        pd_type_pad[:, 2] = type_
        pds.append(pd_type_pad)
    pwd = PairwiseDistance(metric='wasserstein', n_jobs=8)
    distances_type = pwd.fit_transform(np.stack(pds))
    distances = distances + distances_type**2
distances.shape
distances = np.sqrt(distances)
print(time.time() - t)
# %%
np.savetxt(join('graph_data', dataset_name, 'distances_' + dataset_name +
                '.txt'), distances)
# %%
distances = np.loadtxt(join('graph_data', dataset_name,
                            'distances_' + dataset_name + '.txt'))

# %%
matshow(distances)
plt.show()
# %%
D = distances
D = 0.5 * (D + D.transpose())

dim = 3
model = MDS(n_components=dim, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(D)

if dim == 2:
    plt.scatter(out[:, 0], out[:, 1], c=y)
    plt.axis('equal')
    plt.title("MDS 2D using bottleneck of PD for " + dataset_name)
    plt.show()
if dim == 3:
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(out[:, 0], out[:, 1], out[:, 2], c=y)
    plt.title("MDS 3D using bottleneck of PD for " + dataset_name)
    pyplot.show()
# %%

out_pd = pd.DataFrame(out)
out_pd['label'] = y
fig = px.scatter_3d(out_pd, x=0, y=1, z=2,
                    color='label')
fig.show()
# %%
