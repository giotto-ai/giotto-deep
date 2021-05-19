# %%
# flake8: noqa E402

from typing import Union, Optional, List, Dict
import torch
from torch import nn
from gtda.homology import VietorisRipsPersistence as vrp  # type: ignore
import plotly.express as px  # type: ignore
import plotly.figure_factory as ff  # type: ignore
import pandas as pd  # type: ignore
from itertools import chain, combinations
import numpy as np  # type: ignore
from tqdm import tqdm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from transformers import AutoConfig, AutoModelForSequenceClassification  # type: ignore
from transformers import AutoTokenizer  # type: ignore
import networkx as nx  # type: ignore
from matplotlib.patches import ArrowStyle  # type: ignore
import bezier  # type: ignore
from time import sleep
# to use: Path.MOVETO etc. see https://matplotlib.org/stable/_modules/matplotlib/path.html
from matplotlib.path import Path # type: ignore

# rotating vectors
from scipy.spatial.transform import Rotation  # type: ignore

from gtda.homology import FlagserPersistence as flp

import gdeep
import importlib
importlib.reload(gdeep)
from gdeep.optimisation import PersistenceGradient


# %%
# n = 11
# hom_dim = (0, 1, 2, 3)

# dist = torch.rand((n, n)) + 1  # simulate the weighted directed graph
# dist = dist * (torch.ones(n, n) - torch.eye(n, n))
# dist_arr = dist.detach().numpy().copy()
# pg = PersistenceGradient(homology_dimensions=hom_dim,
#                          zeta=0.00,
#                          collapse_edges=False,
#                          metric="precomputed",
#                          directed=True)

# fp = flp(homology_dimensions=hom_dim)
# fp.fit_transform_plot([dist_arr])
# %%
# ArrowStyle with an arrow in the middle
class CurveMiddle(ArrowStyle._Curve):
    """An arrow with a head at its center.
    Custom Arrow style for matplotlib that is not part of matplotlib."""
    
    def __init__(self, head_length=.4, head_width=.2):
        """
        Parameters
        ----------
        head_length : float, default: 0.4
            Length of the arrow head.

        head_width : float, default: 0.2
            Width of the arrow head.
        """
        super().__init__(beginarrow=False, endarrow=True,
                                head_length=head_length,
                                head_width=head_width)

    @staticmethod
    def angle_between(v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2':

        Args:
            v1 (np.array): n-dimensional vector
            v2 (np.array): n-dimensional vector

        Returns:
            (float): angle between vectors 'v1' and 'v2' in radians.
        """
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def transmute(self, path, mutation_size, linewidth):
        _path, _fillable = super().transmute(path, mutation_size, linewidth)
        
        bezier_curve = bezier.Curve(path.vertices.T, degree=2)
        midpoint = bezier_curve.evaluate(0.5)
        midpoint_direction = bezier_curve.evaluate_hodograph(0.5)  # shape (2, 1)

        # translate arrow
        if(len(_path) > 1):
            _path[1].vertices += midpoint.transpose() - _path[1].vertices[1, :]
            
            # rotate arrow
            rotation_angle =  np.pi/2 - self.angle_between(midpoint_direction[:, 0],
                                        _path[1].vertices[0] - _path[1].vertices[-1])
            c, s = np.cos(rotation_angle), np.sin(rotation_angle)
            rotation_matrix = np.array(((c, -s), (s, c)))
            for i in [0, -1]:
                # vector connecting the two endpoints of the arrow
                arrow_side = _path[1].vertices[i] - _path[1].vertices[1]
                # rotate arrow side
                _path[1].vertices[i] += (np.einsum('kj, j-> k',
                                                   rotation_matrix, arrow_side)
                                        - arrow_side)
        return _path, _fillable

# %%
# Debugging Arrow style class

G = nx.DiGraph()
G.add_nodes_from(['a', 'b'])
G.add_edge('a', 'b')

nx.draw(G, arrowstyle=CurveMiddle(head_length=1.6, head_width=.8),
        connectionstyle = "arc3,rad=0.0",
        with_labels=True)
plt.show()

nx.draw(G, arrowstyle=CurveMiddle(head_length=1.6, head_width=.8),
        connectionstyle = "arc3,rad=0.8",
        with_labels=True)
plt.show()

nx.draw(G, arrowstyle=CurveMiddle(head_length=1.6, head_width=.8),
        connectionstyle = "arc3,rad=0.1",
        with_labels=True)
plt.show()
# %%


def plot_weighted_graph(weight_matrix: Union[np.ndarray, torch.Tensor],*,
                        vertex_labels: Optional[List[str]] = None,
                        save_path: Optional[str] = None) -> None:
    """Plot the graph of a weighted graph. The width of the edges
    correspond to the weights.

    Args:
        weight_matrix (Union[np.ndarray, torch.Tensor]): weight matrix of
            the graph.
        vertex_labels (Optional[List[str]]): List of vertex labels. If `None` only the
            edges
        save_path (Optional[str]): path to save the graph plot. If `None` the graph is
            plotted.
    """
    if type(weight_matrix) == torch.Tensor:
        weight_matrix = weight_matrix.detach().numpy().copy()

    G = nx.from_numpy_matrix(weight_matrix, create_using=nx.DiGraph)
    layout = nx.spring_layout(G)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    pos = nx.kamada_kawai_layout(G)  # graph visualization in a force driven way

    connectionstyle = "arc3,rad=0.1"



    if vertex_labels is None:
        options = {
            "node_color": "blue",
            "node_size": 5,
            "connectionstyle": connectionstyle,
            "with_labels":False,
            "arrowstyle": CurveMiddle(),
        }
        nx.draw_networkx(G, pos=pos, width=weights, **options)
        # nx.draw_networkx_edges(G, pos,
        #                         connectionstyle=connectionstyle,
        #                         width=weights, arrowstyle='-')
    else:
        nx.draw(G, pos=pos, width=weights)
        nx.draw_networkx_labels(G, pos, layout, font_size=10,
                                connectionstyle=connectionstyle,
                                width=weights, #rrowstyle='-|>',
                                font_family='sans-serif')
    if save_path is None:
        plt.show()
    else:
        plt.ioff()
        plt.savefig(save_path, format="svg")

# %%
sample_attention_graphs = np.load(
    "attention_graphs_arrays/attention_scores_bert_sample_66.npy")
# uncomment to hide [CLS] and [SEP] token
# sample_attention_graphs = sample_attention_graphs[: ,: ,1:-1, 1:-1]


#print("input sentence:\n", "Believe it or not, this was at one time the worst movie I had ever seen. Since that time, I have seen many more movies that are worse (how is it possible??) Therefore, to be fair, I had to give this movie a 2 out of 10. But it was a tough call.")
#print("attention_graphs shape:", sample_attention_graphs.shape)

for layer, head in [(1, 1)]:#, (3, 1), (8,3), (16, 3), (23, 1)]:
    plot_weighted_graph(sample_attention_graphs[layer, head],
                        save_path=("attention_graph_plots/" +
                                   f"attention_graph_{layer}_{head}.svg"))# %%

# %%

# %%
