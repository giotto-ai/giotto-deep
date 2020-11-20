#%%
# Create binary data cloud

%reload_ext autoreload
%autoreload 2
%matplotlib inline

seed=42

import numpy as np
from sklearn import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F


from gtda.plotting import plot_point_cloud

import matplotlib.pyplot as plt
#from scipy.spatial.distance import KDTree
# %%
data, label = datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=seed)

A = data[label==0]
B = data[label==1] + [2,0]
# %%
plot_point_cloud(np.concatenate((A, B)))
# %%
n_samples = 100
sample_points = np.random.rand(n_samples, 2)

sample_points = sample_points.dot(np.diag([4,2])) + np.array([-1,-1])
# %%
plot_point_cloud(np.concatenate((A, B,sample_points)))
# %%
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1])
plt.scatter(sample_points[:,0],sample_points[:,1])

plt.show()
# %%
# Train a simple logistic regression model for the binary classification task



log_reg = LogisticRegressionNN(2)

X_train, y_train = torch.from_numpy(data).float(), torch.from_numpy(label).long()

