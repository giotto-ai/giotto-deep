from gdeep.trainer import Trainer
import os
import copy
import time
from functools import wraps
import warnings
from typing import Tuple, Optional, Callable, Any, Dict, List, Type, Union
import torch.nn.functional as f
from torch.optim import Optimizer
import torch
from optuna.trial._base import BaseTrial
import numpy as np
from tqdm import tqdm
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import optuna
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler  # noqa
from ..utility.optimization import MissingClosureError
from gdeep.models import ModelExtractor
from gdeep.utility import _inner_refactor_scalars
from gdeep.utility import DEVICE
from .metrics import accuracy
from gdeep.utility.custom_types import Tensor
from abc import ABCMeta, abstractmethod
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KDTree
from gudhi import SimplexTree
import numpy as np
from gtda.homology import VietorisRipsPersistence
import itertools
import matplotlib.pyplot as plt
from gtda.plotting import plot_point_cloud
from gtda.local_homology import KNeighborsLocalVietorisRips, RadiusLocalVietorisRips
from gtda.diagrams import PersistenceEntropy
from sklearn.preprocessing import FunctionTransformer
from gtda.pipeline import make_pipeline


def local_homology_preprocess(
    X: Tensor,
    n_neighbors: Tuple[int, int],
    homology_dimensions: Tuple[int, ...],
    dim: int,
) -> np.ndarray:
    """
    A helper file for topological regularizer
    Computes the local homology, used for the linear surrogate function
    Args:
        X: The point cloud on which the surrogate function is evaluated
        n_neighbors:
            The sizes of the KNN-neighborhoods to be considered on the first
            and the second pass, respectively. An argument for
            ``gtda.local_homology.KNeighborsLocalVietorisRips``
        homology_dimensions: what homology dimensions are computed
        dim: the homology dimension based on which the neighbors are computed
             In the case of the topological regularizer, this is 1.
    Returns
    -------
        conns : An n by 2 numpy array describing the neighbors
    """
    kn_lh = KNeighborsLocalVietorisRips(
        n_neighbors=n_neighbors, homology_dimensions=homology_dimensions
    )
    mod_pe = make_pipeline(
        PersistenceEntropy(), FunctionTransformer(func=lambda X: 2**X)
    )
    pipe = make_pipeline(kn_lh, mod_pe)
    loc_dim_features = pipe.fit_transform(X)
    dimension = dim
    dim_index = homology_dimensions.index(dimension)
    colors = loc_dim_features[:, dim_index]
    cutoffs = np.round(colors, 0)
    nbrs = kn_lh.relevant_neighbors_.kneighbors()[1]
    connections = []
    for i in range(len(colors)):
        cutoff = int(cutoffs[i])
        trick = np.repeat(i, cutoff)
        tmp_neighbors = nbrs[i]
        tmp_trick = np.stack((nbrs[i][0:cutoff], trick))
        maxes = np.amax(tmp_trick, 0)
        mins = np.amin(tmp_trick, 0)
        for j in range(len(maxes)):
            connections.append([mins[j], maxes[j]])
    conns = np.array(unique_list(connections))
    return conns


def create_dummy_loader(
    tensor_x_t: Tensor,
) -> DataLoader[Tuple[Tensor, ...]]:
    """
    A helper file for topological regularizer
    A function for formatting the data for evaluating
    the linear surrogate function.
    Args:
        tensor_x_t: Torch tensor representing the nodes of
        the linear surrogate function.
    Returns
    -------
        dummy_loader : A torch dataloader based off of tensor_x_t
    """
    dummy = np.zeros(len(tensor_x_t))
    dummy_ind = torch.from_numpy(dummy)
    dummy_set = TensorDataset(tensor_x_t, dummy_ind)
    dummy_loader = DataLoader(dummy_set, batch_size=4)
    return dummy_loader


def unique_list(a_list: list) -> list:
    """
    A helper file for topological regularizer
    Given a list a_list,
    returns the unique elements in that.
    Used for computing the connections
    when building the linear surrogate function
    Args:
         a_list: a list
    Returns
    -------
        uniquelist : a list of unique entries of the list a_list
    """
    uniquelist = []
    used = set()
    for item in a_list:
        tmp = repr(item)
        if tmp not in used:
            used.add(tmp)
            uniquelist.append(item)
    return uniquelist


def evaluate_model_on_grid(
    dummy_loader: DataLoader[Tuple[Tensor, ...]],
    model: torch.nn.Module,
) -> np.ndarray:
    """
    A helper file for topological regularizer
    Evaluates a binary classifier neural netowrk
    on a dataset and returns the class 1 probabilities. This is used for
    computing the persistence of the linear surrogate function.
    Args:
         dummy_loader: Torch dataloader representing
             the knots of the linear surrogate function.
             The response variable can be anything.
         model: Torch model
    Returns
    -------
        results : Class 1 probabilities of the model evaluated on
            the dummy_loader
    """
    model.eval()
    featdim = next(iter(dummy_loader))[0].shape[1]
    # featdim = dummy_loader.dataset.tensors[0].shape[1]
    preds = []
    for i, data in enumerate(dummy_loader, 0):
        inputs, labels = data
        inputs = torch.reshape(inputs, (-1, featdim)).float()
        outputs = model(inputs)
        preds.append(F.softmax(outputs, 1)[:, 1])
    tmp = [p.detach().numpy() for p in preds]
    results = np.concatenate(tmp, axis=0)
    return results


def get_persistent_pairs(
    results: np.ndarray, connections: np.ndarray, expansion: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A helper file for topological regularizer
    Given Class 1 probabilities and node connectivity information,
    returns the persistent pairs of the function
    associated to filtering according to the numerical values
    Args:
        results: an n-vector of class 1 probabilities
        connections: an n by 2 numpy array describing pairwise connections
        among the entries of results
        expansion: The size of the clique expansion when computing homology
    Returns
    -------
        out_pers: filtration values of the critical pairs
        persts: the indices of the nodes corresponding to out_pers
    """
    st = SimplexTree()
    for t in range(len(results)):
        st.insert([t], filtration=results[t])
    for i in range(connections.shape[0]):
        in1 = connections[i, 0]
        in2 = connections[i, 1]
        fil = max(results[in1], results[in2])
        st.insert([in1, in2], filtration=fil)
    st.expansion(expansion)
    st.compute_persistence()
    tmp = st.persistence_pairs()
    persistences = np.zeros([len(tmp), 2])
    pers_indices = np.zeros([len(tmp), 2])
    odd_one_out = 0
    for i in range(len(tmp)):
        inds1 = tmp[i][0]
        inds2 = tmp[i][1]
        if len(inds2) == 0:
            odd_one_out = i
            continue
        for index in range(len(inds1)):
            value = results[inds1[index]]
            if value > persistences[i, 0]:
                persistences[i, 0] = value
                pers_indices[i, 0] = inds1[index]
        for index in range(len(inds2)):
            value = results[inds2[index]]
            if value > persistences[i, 1]:
                persistences[i, 1] = value
                pers_indices[i, 1] = inds2[index]
    out_pers = np.delete(persistences, odd_one_out, axis=0)
    out_pers_ind = np.delete(pers_indices, odd_one_out, axis=0)
    persts = out_pers_ind.astype(int)
    return (out_pers, persts)


def compute_critical_points(
    net: torch.nn.Module,
    coords: Tensor,
    dummy_loader: DataLoader[Tuple[Tensor, ...]],
    connections: np.ndarray,
    expansion: int,
    ind: int = 0,
) -> Tensor:
    """
    A helper file for topological regularizer
    Given a neural network and a dataset, computes the
    ephemeral critical values and points of the linear
    Morse function defined by the dataset and connections.
    The filtering directions is specified by ind (0: Positive, 1 negative)
    A helper file for topological regularizer
    Args:
        model: torch model
        coords: torch.tensor, the nodes of the linear surrogate function
        dummy_loader: torch dataloader containing the coords.
            connections: numpy_array. An n by 2 numpy array describing
            pairwise connections between the nodes.
        expansion: The size of the clique expansion
        ind: integer (for now boolean) describing filtration direction
            0: positive filtration, 1: negative filtration
    Returns
    -------
        coordinates: The coordinates of the critical points that
        contribute to the complexity of the decision boundary.
    """
    res = evaluate_model_on_grid(dummy_loader, net)
    if ind == 1:
        res = 1 - res
    out_pers, persts = get_persistent_pairs(res, connections, expansion)

    """ Get only the persistent pairs that contribute to
    the complexity of the decision boundary:
    These are the ones that are on the different side of the boundary
    """
    tmp = out_pers - 0.5
    cuts = (tmp[:, 0] * tmp[:, 1]) < 0

    goodinds = np.argmin(abs(tmp[cuts, :]), 1)
    tmp_inds = persts[cuts, :]
    indices = np.zeros(tmp_inds.shape[0])
    for i in range(tmp_inds.shape[0]):
        sele = goodinds[i]
        indices[i] = tmp_inds[i, sele]
    # The good values
    indices = indices.astype(int)
    coordinates = coords[indices, :].clone().detach().requires_grad_(True)
    return coordinates


class Regularizer:
    def __init__(self, **kwargs):
        self.regularizer_arguments = kwargs

    """
    An abstract class for handling various regularization schemes.
    A Regularizer needs to have 3 methods:
    - preprocess
    - update_params
    - regularization_penalty
    """

    @abstractmethod
    def preprocess(self, **regularizer_arguments: Any) -> Any:
        """
        performs preprocessing for the regularizer, if any is necessary

        """
        return regularizer_arguments

    @abstractmethod
    def update_params(self, pre_processed_arguments: Any) -> Any:
        """
        Any processing needed in a forward pass of the network
        Anything that doesn't require gradients can go here
        """
        return pre_processed_arguments

    @abstractmethod
    def regularization_penalty(
        self, forward_passed_arguments: Any, model: torch.nn.Module
    ) -> Tensor:
        """
        Computes the actual regression penalty.
        This must be an auto-differentiable function of its inputs
        """
        return torch.sum(forward_passed_arguments) ** 2


class TopologicalRegularizer(Regularizer):
    """
    A topological regularizer for a 2-class classifier
    """

    def preprocess(self, **kwargs):
        """
        Initializes the topological regularizer:
        kwargs:
            n_neighbors: sizes of the KNN neighbors used for computing the local homology
            hom_dim: the homologies to be computed
            expansion:  The size of the clique-expansion when computing homology
            regularization_dataset: Tensor describing where the linear surrogate function is evaluated
        """

        defaultKwargs = {
            "n_neighbors": (6, 20),
            "hom_dim": (0, 1, 2, 3, 4, 5, 6),
            "expansion": 6,
        }
        self.regularizer_arguments = {**defaultKwargs, **kwargs}
        print("Preprocessing regularizer")
        dummy_loader = create_dummy_loader(
            self.regularizer_arguments["regularization_dataset"]
        )
        connections = local_homology_preprocess(
            self.regularizer_arguments["regularization_dataset"],
            self.regularizer_arguments["n_neighbors"],
            self.regularizer_arguments["hom_dim"],
            dim=1,
        )
        self.regularizer_arguments["coords"] = self.regularizer_arguments[
            "regularization_dataset"
        ]
        self.regularizer_arguments["connections"] = connections
        self.regularizer_arguments["dummy_loader"] = dummy_loader
        return True

    def update_params(self, regularizer_arguments):
        """
        Finds the critical vertices of the linear surrogate function
        """
        network = self.regularizer_arguments["model"]
        coords = self.regularizer_arguments["coords"]
        dummy_loader = self.regularizer_arguments["dummy_loader"]
        connections = self.regularizer_arguments["connections"]
        expansion = self.regularizer_arguments["expansion"]
        cpoints1 = compute_critical_points(
            network, coords, dummy_loader, connections, expansion, ind=0
        )
        cpoints2 = compute_critical_points(
            network, coords, dummy_loader, connections, expansion, ind=1
        )
        cpoints = np.concatenate(
            [cpoints1.detach().numpy(), cpoints2.detach().numpy()], 0
        )
        cpoints = torch.tensor(cpoints)
        cpoints.requires_grad = True
        return cpoints
        # pass

    def regularization_penalty(self, forward_passed_arguments, model: torch.nn.Module):
        """
        The penalty is the squared sum of
        the function values at the critical points.
        """
        values = model(forward_passed_arguments)
        tmp = torch.max(F.softmax(values, 1) - 0.5, 1)
        return torch.sum((tmp.values) ** 2)


class TihonovRegularizer(Regularizer):
    """
    An example implementation of a p-norm regularizer
    This regularizer penalizes the p-norm (to the pth power) of the network weights.
    For example, p=1 is LASSO and p=2 is the Ridge Regression.
    reg_params: p (int), (Optional)
    """

    def preprocess(self, **regularizer_arguments: Any):
        defaultKwargs = {"p": 2}
        self.params = {**defaultKwargs, **regularizer_arguments}
        """
        Tihonov-type regularizer needs no preprocessing
        """
        return True

    def update_params(self, regularizer_arguments: Any):
        """
        Tihonov regularizer needs no non-differentiable updates
        """
        return regularizer_arguments

    def regularization_penalty(
        self, forward_passed_arguments: Any, model: torch.nn.Module
    ) -> Tensor:
        """
        The regularization penalty is sum of absolute values of the
        regression coefficients, i.e. network weights
        """
        total = torch.tensor(0)  # , dtype=float)
        for parameter in model.parameters():
            total = total + torch.norm(parameter, self.params["p"]) ** self.params["p"]
        return total


class RegularizedTrainer(Trainer):
    def __init__(self, regularizer, reg_params={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regularizer = regularizer
        self.regularize = True
        self.reg_params = regularizer.preprocess(**reg_params)
        """
        A wrapper for defining training with regularization.
        Automatically preprocesses the regularization parameters and sets regularize=True
        to invoke the regularization logic in Trainer
        """
