#import os

import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd

import torch
from sklearn.decomposition import PCA


def plot_decision_boundary(data, labels, boundary_points, n_components=2, show=True):
    """Plot decision boundaries with the data

    Args:
    -----
     - data 2darray: arraa of the dataset
     - boundary_points 2darray: output of gradient_flow
     - labels list or 1darray: [description]
     - n_components (int, optional): [description]. Defaults to 2.
    """
    
    if len(data[0])>n_components:
        n_comp = n_components
        pca = PCA(n_components=n_comp)
        pca.fit(data)
        X_pca = pca.transform(data)
        bdry_pca = pca.transform(boundary_points)
    else:
        n_comp = len(data[0])
        X_pca = data
        bdry_pca = boundary_points

    df_data = pd.DataFrame(data, columns = ["x"+str(i) for i in range(len(X_pca[0]))])
    df_bdry = pd.DataFrame(bdry_pca, columns = ["z"+str(i) for i in range(len(X_pca[0]))])
    if n_comp == 1:
        fig = px.scatter(df_data,x="x0",color=labels)
        fig2 = px.scatter(df_bdry, x="z0")
        fig.add_trace(fig2.data[0])
        fig.show()
    elif n_comp == 2:
        df_bdry['labels']=[0.6]*df_bdry.shape[0]
        fig = px.scatter(df_data,x="x0",y="x1",color=labels)
        fig2 = px.scatter(df_bdry, x="z0",y="z1",color="labels")
        fig.add_trace(fig2.data[0])
        if show:
            fig.show()
        else:
            return fig
    elif n_comp == 3:
        fig = px.scatter_3d(df_data,x="x0",y="x1",z="x2",color=labels)
        fig2 = px.scatter_3d(df_bdry, x="z0",y="z1",z="z2")
        fig.add_trace(fig2.data[0])
        if show:
            fig.show()
        else:
            return fig
    else:
        fig = px.scatter_3d(df_data,x="x0",y="x1",z="x2",color=labels)
        fig2 = px.scatter_3d(df_bdry, x="z0",y="z1",z="z2")
        fig.add_trace(fig2.data[0])
        fig.show()


def plot_activation_contours(model,delta=0.1, boundary_tuple=((-1.5, 1.5),(-1.5, 1.5))):
    """Plot the contours of the last layer softmax
        
        Args:
        -----
        - data 2darray: arraa of the dataset
        - boundary_points 2darray: output of gradient_flow
        - labels list or 1darray: [description]
        - n_components (int, optional): [description]. Defaults to 2.
        """
    
    delta = delta
    x = np.arange(*boundary_tuple[0], delta)
    y = np.arange(*boundary_tuple[1], delta)
    X, Y = np.meshgrid(x, y)

    X_tensor, Y_tensor = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    X_tensor = X_tensor.reshape((X_tensor.shape[0],X_tensor.shape[1],1))
    Y_tensor = Y_tensor.reshape((X_tensor.shape[0],X_tensor.shape[1],1))
    XY_tensor = torch.cat((X_tensor,Y_tensor), 2)

    XY_tensor = XY_tensor.reshape((-1,2))

    Z_tensor = model.forward(None, XY_tensor)
    Z_tensor = Z_tensor[:,0].reshape((X_tensor.shape[0],X_tensor.shape[1]))
    Z = Z_tensor.detach().numpy()
    fig = go.Figure(data =go.Contour(z=Z,x=x,y=y))
    fig.show()
