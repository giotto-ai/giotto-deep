#from gdeep.trainer import Trainer
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
from dataclasses import dataclass


def _local_homology_preprocess(
    X: Tensor,
    n_neighbors: Tuple[int, int],
    homology_dimensions: Tuple[int, ...],
    dim: int,
) -> np.ndarray:
    """
    A helper file for topological regularizer
    Computes the local homology, used for the linear surrogate function
    Args:
        X:
            The point cloud on which the surrogate function is evaluated
        n_neighbors:
            The sizes of the KNN-neighborhoods to be considered on the first
            and the second pass, respectively. An argument for
            ``gtda.local_homology.KNeighborsLocalVietorisRips``
        homology_dimensions:
            what homology dimensions are computed
        dim:
            the homology dimension based on which the neighbors are computed
            In the case of the topological regularizer, this is 1.
    Returns:
        conns:
            An n by 2 numpy array describing the neighbors
    """
    kn_lh = KNeighborsLocalVietorisRips(
        n_neighbors=n_neighbors, homology_dimensions=homology_dimensions
    )
    mod_pe = make_pipeline(
        PersistenceEntropy(), FunctionTransformer(func=lambda X: 2 ** X)
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
    conns = np.array(_unique_list(connections))
    return conns


def _create_dummy_loader(
    tensor_x_t: Tensor,
) -> DataLoader[Tuple[Tensor, ...]]:
    """
    A helper file for topological regularizer
    A function for formatting the data for evaluating
    the linear surrogate function.
    Args:
        tensor_x_t:
            torch tensor representing the nodes of
            the linear surrogate function.
    Returns:
        dummy_loader:
            a torch dataloader based off of tensor_x_t
    """
    dummy = np.zeros(len(tensor_x_t))
    dummy_ind = torch.from_numpy(dummy)
    dummy_set = TensorDataset(tensor_x_t, dummy_ind)
    dummy_loader = DataLoader(dummy_set, batch_size=4)
    return dummy_loader


def _unique_list(a_list: list) -> list:
    """
    A helper file for topological regularizer
    Given a list a_list,
    returns the unique elements in that.
    Used for computing the connections
    when building the linear surrogate function
    Args:
         a_list:
             a list
    Returns
        uniquelist:
            a list of unique entries of the list a_list
    """
    uniquelist = []
    used = set()
    for item in a_list:
        tmp = repr(item)
        if tmp not in used:
            used.add(tmp)
            uniquelist.append(item)
    return uniquelist


def _evaluate_model_on_grid(
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
    Returns:
        results : Class 1 probabilities of the model evaluated on
            the dummy_loader
    """
    model.eval()
    featdim = next(iter(dummy_loader))[0].shape[1]
    preds = []
    for i, data in enumerate(dummy_loader, 0):
        inputs, labels = data
        inputs = torch.reshape(inputs, (-1, featdim)).float()
        outputs = model(inputs)
        preds.append(F.softmax(outputs, 1)[:, 1])
    tmp = [p.detach().numpy() for p in preds]
    results = np.concatenate(tmp, axis=0)
    return results


def _get_persistent_pairs(
    results: np.ndarray, connections: np.ndarray, expansion: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A helper file for topological regularizer
    Given Class 1 probabilities and node connectivity information,
    returns the persistent pairs of the function
    associated to filtering according to the numerical values
    Args:
        results:
            an n-vector of class 1 probabilities
        connections:
            an n by 2 numpy array describing pairwise connections
            among the entries of results
        expansion:
            The size of the clique expansion when computing homology
    Returns:
        persistences:
            filtration values of the critical pairs
        pers_indices:
            the indices of the nodes corresponding to out_pers
    """
    st = SimplexTree()
    # insert vertices
    for t in range(len(results)):
        st.insert([t], filtration=results[t])
    # insert edges
    for i in range(connections.shape[0]):
        index1 = connections[i, 0]
        index2 = connections[i, 1]
        fil = max(results[index1], results[index2])
        st.insert([index1, index2], filtration=fil)
    # do the clique expansion
    st.expansion(expansion)
    st.compute_persistence()
    tmp = st.persistence_pairs()  # returns pairs of simplices
    persistences = np.zeros([len(tmp), 2])
    pers_indices = np.zeros([len(tmp), 2])
    # next we loop through the persistent pairs
    # to see which vertices are responsible for the births and deaths
    for i in range(len(tmp)):
        birth_inds = tmp[i][0]
        death_inds = tmp[i][1]
        # omit the infinitely persistent features, these are just
        # for those we can just leave everything to 0
        # recall f maps to [0,1]
        if len(death_inds) == 0:
            continue
        # the birth (death) time of the simplex is the maximal
        # birth (death) time of its simplices:
        for index in range(len(birth_inds)):
            putative_birth_time = results[birth_inds[index]]
            if putative_birth_time > persistences[i, 0]:
                persistences[i, 0] = putative_birth_time
                pers_indices[i, 0] = birth_inds[index]
        for putative_death_time in range(len(death_inds)):
            putative_death_time = results[death_inds[index]]
            if putative_death_time > persistences[i, 1]:
                persistences[i, 1] = putative_death_time
                pers_indices[i, 1] = death_inds[index]
    pers_indices = pers_indices.astype(int)
    return (persistences, pers_indices)


def _compute_critical_points(
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
        model:
            torch model
        coords:
            torch.tensor, the nodes of the linear surrogate function
        dummy_loader:
            torch dataloader containing the coords.
            connections: numpy_array. An n by 2 numpy array describing
            pairwise connections between the nodes.
        expansion:
            The size of the clique expansion
        ind:
            integer (for now boolean) describing filtration direction
            0: positive filtration, 1: negative filtration
    Returns:
        coordinates:
            The coordinates of the critical points that
            contribute to the complexity of the decision boundary.
    """
    res = _evaluate_model_on_grid(dummy_loader, net)
    if ind == 1:  # negative filtration 'class 0 probability'
        res = 1 - res
    # Get the persistent pairs of the linear surrogate function
    critical_values, indices_of_critical_points = _get_persistent_pairs(
        res, connections, expansion
    )

    """ Next we filter out the persistent pairs that contribute to
    the complexity of the decision boundary:
    These are the ones that are active at the db: so born before it and die after it.
    Recall f: X -> [0,1], maps data to class 1 probability (class 0 probability)
    
    In that case the decision boundary: f^{-1}(1/2)
    
    We can simplify the boundary either by pushing the homology generator above it (delay its birth)
    or pushing the annihilator below it (expedite its death). We should do whichever is more economical,
    i.e., closer to the cutoff (1/2). Next we do just that.
    
    In the following code variable name containing 'rel_inds' is used to refer to indices with relative to
    the critical points. (i.e. rel_inds = 0 is the first critical pair)
    'inds' without this qualifier are indices with respect to the coordinates (inds = 0: first vertex of coords)
    """
    # signed distance of the critical values (of the critical pairs) from the db:
    signed_distance_from_db = critical_values - 0.5
    # pairs of critical points on different sides of the db: i.e. signed distance XOR negative (or equiv. product<0)
    db_pairs_rel_inds = (
        signed_distance_from_db[:, 0] * signed_distance_from_db[:, 1]
    ) < 0  # column XOR Negative
    # for each pair of critical points, decide if birth or death is closer to the db:
    columns = np.argmin(
        abs(signed_distance_from_db[db_pairs_rel_inds, :]), 1
    )  # argmin of columns
    # the indices of the vertices that are on either side of db, expressed in original coords indices:
    db_pair_inds = indices_of_critical_points[db_pairs_rel_inds, :]
    # Next we loop through the selected pairs to see which one we push: birth or death, and record the index of that vertex:
    indices = np.zeros(db_pair_inds.shape[0])
    for i in range(db_pair_inds.shape[0]):
        column = columns[i]  #:0 birth, 1: death
        indices[i] = db_pair_inds[
            i, column
        ]  # pick the index of the coordinate accordingly
    # indices of the selected critical points, in reference to the original coords:
    indices = indices.astype(int)
    # return the coordinates, directly digestible by the model
    coordinates = coords[indices, :].clone().detach().requires_grad_(True)
    return coordinates


# class Regularizer:
#    """
#    An abstract class for handling various regularization schemes.
#    Args:
#        lamda:
#            float the regression penalty coefficient that is typically present in regularization
#    """

#    def __init__(self, lamda: float = 1, **kwargs):
#        self.lamda = lamda
#        self.preprocess(**kwargs)

#    @abstractmethod
#    def preprocess(self):
#        """
#        performs preprocessing for the regularizer, if any is necessary

#        """
#        pass

#    @abstractmethod
#    def regularization_penalty(
#        self, model: torch.nn.Module
#    ) -> Tensor:  # , pre_processed_arguments: Any) -> Any:
#        """
#        Any processing needed in a forward pass of the network
#        Anything that doesn't require gradients can go here
#        """
#        pass


@dataclass
class TopologicalRegularizerData:
    """
    Container for the data needed for TopologicalRegularizer.
    Args:
        regularization_dataset:
            tensor describing where the linear surrogate function is evaluated
        n_neighbors:
            sizes of the KNN neighbors used for computing the local homology
        hom_dim:
            the homologies to be computed
        expansion:
            the size of the clique-expansion when computing homology
    """

    regularization_dataset: Tensor
    n_neighbors: Tuple[int, int] = (6, 20)
    hom_dim: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    expansion: int = 6

    def __post_init__(self):
        self.dummy_loader = _create_dummy_loader(self.regularization_dataset)
        self.connections = _local_homology_preprocess(
            self.regularization_dataset, self.n_neighbors, self.hom_dim, dim=1
        )


from typing import Protocol


class Regularizer(Protocol):
    def regularization_penalty(self, model) -> Tensor:
        pass


class TopologicalRegularizer:
    """
    A topological regularizer for a 2-class classifier
    """

    def __init__(self, lamda, data: TopologicalRegularizerData):
        self.lamda = lamda
        self.data = data

    def regularization_penalty(self, model: torch.nn.Module) -> Tensor:
        """
        The penalty is the squared sum of
        the function values at the critical points.
        """
        network = model  # trainer.model
        coords = self.data.regularization_dataset
        dummy_loader = self.data.dummy_loader
        connections = self.data.connections
        expansion = self.data.expansion
        cpoints1 = _compute_critical_points(
            model, coords, dummy_loader, connections, expansion, ind=0
        )
        cpoints2 = _compute_critical_points(
            model, coords, dummy_loader, connections, expansion, ind=1
        )
        cpoints = np.concatenate(
            [cpoints1.detach().numpy(), cpoints2.detach().numpy()], 0
        )
        cpoints = torch.tensor(cpoints)
        cpoints.requires_grad = True
        values = model(cpoints)
        tmp = torch.max(F.softmax(values, 1) - 0.5, 1)
        return self.lamda * torch.sum((tmp.values) ** 2)


class TihonovRegularizer:
    """
    An example implementation of a p-norm regularizer
    This regularizer penalizes the p-norm (to the pth power) of the network weights.
    For example, p=1 is LASSO and p=2 is the Ridge Regression.
    reg_params: p (int), (Optional)
    """

    def __init__(self, lamda: float, p: int):
        self.lamda = lamda
        self.p = p

    def regularization_penalty(self, model: torch.nn.Module) -> Tensor:
        """
        The regularization penalty is sum of the pth powers of the p-norms of the
        regression coefficients, i.e. network weights
        """
        total = torch.tensor(0)
        for parameter in model.parameters():
            total = total + torch.norm(parameter, self.p) ** self.p
        return self.lamda * total
