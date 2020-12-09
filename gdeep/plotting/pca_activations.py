import plotly.express as px
import pandas as pd

import torch
from sklearn.decomposition import PCA

from gdeep.create_nets.utility import get_activations


def plot_PCA_activations(model, X, labels, n_components=2):
    """Plot PCA of the activations of all layers of the neural network

    Args:
    -----
     - model ([type]): [description]
     - X 2darray: [description]
     - labels list or 1darray: [description]
     - n_components (int, optional): [description]. Defaults to 2.
    """
    activations_layers = get_activations(model,torch.from_numpy(X).float())
    
    for activations_layer in activations_layers.get_outputs():
        if activations_layer.shape[1]>n_components:
            n_comp = n_components
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(activations_layer.cpu())
        else:
            n_comp = activations_layer.shape[1]
            X_pca = activations_layer

        df = pd.DataFrame(X_pca, columns = ["x"+str(i) for i in range(len(X_pca[0]))])
        if n_comp == 1:
            fig = px.scatter(df,x="x0",color=labels)
            fig.show()
        elif n_comp == 2:
            fig = px.scatter(df,x="x0",y="x1",color=labels)
            fig.show()
        elif n_comp == 3:
            fig = px.scatter_3d(df,x="x0",y="x1",z="x2",color=labels)
            fig.show()
        else:
            fig = px.scatter_3d(df,x="x0",y="x1",z="x2",color=labels)
            fig.show()


