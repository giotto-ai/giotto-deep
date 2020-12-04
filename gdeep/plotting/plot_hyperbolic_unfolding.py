import os

import pandas as pd

from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go

from giotto.decision_boundary.hyperbolic_unfolding import Geodesics

class DimensionReducer():
    def __init__(n_components=2):
        self.n_components = n_components

    def fit(dataset: numpy.ndarray):
        self.need_pca = True
        if dataset.shape[-1] <= self.n_components:
            self.need_pca = False
        if self.need_pca:
            self.pca = PCA(n_components=n_components)
            self.pca.fit(data)

    def transform(dataset: numpy.ndarray):
        if self.need_pca = True:
            return self.pca.transform(data)
        else:
            return data

    def fit_transform(dataset: numpy.ndarray):
        self.fit(dataset)
        return self.transform(datset)



class DatasetToDataFrameConverter():
    __init__(self, data):
        return pd.DataFrame(data, columns = ["x"+str(i) for i in range(data.shape[-1])])

class GeodesicPlotter():
    """Plotter template that is able to plot a family of Geodesics
    in different ways

    Args:
        geodesic (Geodesics): family of geodesics
    """
    def __init__(self, geodesic: Geodesics):
        self.geodesic = geodesic
        self.dim = self.geodesic.dim
    def plot_to_fig(self):
        pass
    def plot_to_file(self, filename: str):
        fig = self.plot_to_fig()
        if plot_file:
                try:
                    fig.write_html(os.path.join('plots', plot_file))
                except:
                    os.mkdir('plots')
                    fig.write_html(os.path.join('plots', plot_file))
    def show(self):
        fig = self.plot_to_fig()
        fig.show()

class GeodesicTrajectoryPlotter(GeodesicPlotter):
    """Plotter class that is able to plot the trajectories of
    a family of Geodesics

    Args:
        geodesic (Geodesics): family of geodesics
    """
    def __init__(self, geodesic):
        try:
            assert(geodesic.dim==3)
        except:
            raise ValueError('Only works in 3 dimension up to now!')
        super().__init__(geodesic)

        # TODO rewrite function
        df_trajectories ={}
        for i in range(self.n_geodesics):
            df_trajectories[i] = pd.DataFrame(self.trajectories()[:,i,:],\
                                              columns=["x"+str(i) for i in range(self.dim)])
        if self.dim==2:
            fig = go.Figure()
            for i in range(self.n_geodesics):
                fig.add_trace(go.Scatter(x=df_trajectories[i]["x0"], y=df_trajectories[i]["x1"]))

            fig.update_yaxes(
                    scaleanchor = "x",
                    scaleratio = 1,
                  )

class GeodesicEnpointPlotter(GeodesicPlotter):
    """Plotter class that is able to plot the trajectories of
    a family of Geodesics

    Args:
        geodesic (Geodesics): family of geodesics
    """
    def __init__(self, geodesic):
        try:
            assert(geodesic.dim==2 or geodesic.dim==3)
        except:
            raise ValueError('Only works in 2 and 3 dimension up to now!')
        super().__init__(geodesic)
    def plot_to_fig(self):
        if self.dim==3:
            df_endpoints = pd.DataFrame(self.geodesic.endpoints(), columns=["x"+str(i) for i in range(3)])
            fig = px.scatter_3d(df_endpoints, x="x0", y="x1", z="x2")
            return fig


class DatasetPlotter():
    def __init__(self, dataset: pandas.core.frame.DataFrame):
        try:
            assert(geodesic.dim==2 or geodesic.dim==3)
        except:
            raise ValueError('Only works in 2 and 3 dimension up to now!')
        self.dataset = dataset
    def plot_to_fig(self):
        pass



class PlotCombiner():
    def __init__(self, *args):
        try:
            assert(geodesic.dim==2 or geodesic.dim==3)
        except:
            raise ValueError('Only works in 2 and 3 dimension up to now!')
        self.plotter_list = args
    def plot_to_fig(self):
        fig = go.Figure()
        for plotter in plotter_list:
            fig.add_trace(plotter.plot_to_fig())
        return fig
    def plot_to_file(self, filename: str):
        pass
    def show(self):
        pass