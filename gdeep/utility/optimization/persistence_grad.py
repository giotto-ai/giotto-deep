from functools import reduce
from itertools import chain, combinations
import multiprocessing
from typing import Iterator, Any, Callable, Tuple, List, Optional, Union, Set

import torch
from torch import optim
from gph.python import ripser_parallel
from gtda.homology import FlagserPersistence as Flp
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.graph_objects import Figure
import numpy as np
from tqdm import tqdm
import operator as op

from gdeep.utility import DEVICE

from gdeep.utility.custom_types import Tensor

from gdeep.utility.custom_types import Array


class PersistenceGradient:
    """This class computes the gradient of the persistence
    diagram with respect to a point cloud. The algorithms has
    first been developed in https://arxiv.org/abs/2010.08356 .

    Disclaimer: this algorithm works well for generic point clouds.
    In case your point cloud has many simplices with same
    filtration values, the matching of the points to the persistent
    features may fail to disambiguate.

    Args:
        zeta :
            the relative weight of the regularisation part
            of the ``persistence_function``
        homology_dimensions :
            tuple of homology dimensions
        collapse_edges :
            whether to use Collapse or not. Not implemented yet.
        max_edge_length :
            the maximum edge length
            to be consider not infinity
        approx_digits :
            digits after which to trunc floats for
            list comparison
        metric :
            either ``"euclidean"`` or ``"precomputed"``.
            The second one is in case of ``x`` being
            the pairwise-distance matrix or
            the adjacency matrix of a graph.
        directed :
            whether the input graph is a directed graph
            or not. Relevant only if ``metric = "precomputed"``

    Examples::

        from gdeep.utility.optimization import PersistenceGradient
        # prepare the datum
        X = torch.tensor([[1, 0.], [0, 1.], [2, 2], [2, 1]])
        # select the homology dimensions
        hom_dim = [0, 1]
        # initialise the class
        pg = PersistenceGradient(homology_dimensions=hom_dim,
                                 zeta=0.1,
                                 max_edge_length=3,
                                 collapse_edges=False)
        # run the optimisation for four epochs!
        pg.sgd(X, n_epochs=4, lr=0.4)

    """

    dist_mat: Tensor

    def __init__(
        self,
        homology_dimensions: Optional[List[int]],
        zeta: float = 0.5,
        collapse_edges: bool = False,
        max_edge_length: float = np.inf,
        approx_digits: int = 6,
        metric: str = "euclidean",
        directed: bool = False,
    ) -> None:

        self.collapse_edges = collapse_edges
        self.max_edge_length = max_edge_length
        self.metric = metric
        self.directed = directed
        self.approx_digits = approx_digits
        self.zeta = zeta
        if homology_dimensions:
            self.homology_dimensions = homology_dimensions
        else:
            self.homology_dimensions = [0, 1]

    @staticmethod
    def powerset(iterable: Iterator, max_length: int) -> Iterator:
        """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        up to `max_length`."""
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(0, max_length + 1))

    @staticmethod
    def _parallel_apply_along_axis(
        func1d: Callable, axis: int, arr: Array, *args: Any, **kwargs: Any
    ) -> Array:
        """
        Like ``numpy.apply_along_axis()``, but takes advantage of multiple
        cores.
        """
        # Effective axis where apply_along_axis() will be applied by each
        # worker (any non-zero axis number would work, so as to allow the use
        # of `np.array_split()`, which is only done on axis 0):
        effective_axis = 1 if axis == 0 else axis
        if effective_axis != axis:
            arr = arr.swapaxes(axis, effective_axis)
        n_processes = min(len(arr) // 2, multiprocessing.cpu_count())
        # Chunks for the mapping (only a few chunks):
        chunks = [
            (func1d, effective_axis, sub_arr, args, kwargs)
            for sub_arr in np.array_split(arr, n_processes)
        ]
        pool = multiprocessing.Pool(n_processes)
        individual_results = pool.map(unpacking_apply_along_axis, chunks)  # type: ignore
        # Freeing the workers:
        pool.close()
        pool.join()
        return np.concatenate(individual_results)  # type: ignore

    def _simplicial_pairs_of_indices(self, x: Tensor) -> Tensor:
        """Private function to compute the pair of indices in X to
        matching the simplices.
        """
        simplices = list(
            self.powerset(
                list(range(0, len(x))),  # type: ignore
                max(self.homology_dimensions) + 2,
            )
        )[1:]
        simplices_array = np.array(simplices, dtype=object).reshape(-1, 1)
        comb_number = comb(max(self.homology_dimensions) + 2, 2)
        # the current computation bottleneck
        if len(simplices_array) > 10000000:
            pairs_of_indices = self._parallel_apply_along_axis(
                _combinations_with_single, 1, simplices_array, comb_number
            )
        else:
            pairs_of_indices = np.apply_along_axis(
                _combinations_with_single, 1, simplices_array, comb_number
            )
        return torch.tensor(pairs_of_indices, dtype=torch.int64)

    def phi(self, x: Tensor) -> Tensor:
        """This function is from :math:`(R^d)^n` to :math:`R^{|K|}`,
        where K is the top simplicial complex of the VR filtration.
        It is defined as:
        :math:`\\Phi_{\\sigma}(X)=max_{i,j \\in \\sigma}||x_i-x_j||.`

        Args:
            x:
                the argument of :math:`\\Phi`, a tensor

        Returns:
            Tensor:
                the value of :math:`\\Phi` at ``x``
        """

        if self.metric == "precomputed":
            self.dist_mat = x
        else:
            self.dist_mat = torch.cdist(x, x)
        simplicial_pairs = self._simplicial_pairs_of_indices(x).reshape(-1, 2)
        ks = simplicial_pairs[:, 0]
        js = simplicial_pairs[:, 1]
        comb_number = comb(max(self.homology_dimensions) + 2, 2)
        # morally, this is what one would like to do:
        # lista = [max([dist_mat[pair] for pair in pairs]) for pairs in
        #    simplicial_pairs if max([dist_mat[pair] for pair in pairs])
        #    <= self.max_edge_length]
        lista = torch.max(
            (
                torch.gather(
                    torch.index_select(self.dist_mat, 0, ks), 1, js.reshape(-1, 1)
                )
            ).reshape(-1, comb_number),
            1,
        )[0]
        lista = torch.sort(lista)[0]
        return lista

    @staticmethod
    def _compute_pairs(x: Tensor) -> Tuple[Tensor, Tensor]:
        """Use giotto-tda to compute homology (b,d) pairs

        Args:
            x:
                this is the input point cloud or the input
                pairwise distance or the adjacency matrix of a graph

        Returns:
            Tensor, Tensor:
                the first tensor is the tensor obtained from
                the list of pairs ``(b,d)``; the second 1d tensor
                contains the homology
                group index"""

        dgms = Flp().fit_transform([x.detach().numpy()])
        pairs = dgms[0]
        return pairs[:, :2], pairs[:, 2]

    def _persistence(self, x: Tensor) -> Tuple[Array, Tensor, Tensor]:
        """This function computes the persistence permutation.
        This permutation permutes the filtration values and matches
        them as follows:
        :math:`Pers:Filt_K \\subset \\mathbb R^{|K|} \\to (\\mathbb R^2)^p
        \\times \\mathbb R^q, \\Phi(X) \\mapsto D = \\cup_i^p
        (\\Phi_{\\sigma_i1}(X) ,\\Phi_{\\sigma_i2}(X) )
        \\times \\cup_j^q (\\Phi_{\\sigma_j}(X),+\\infty).`

        Args:
            x:
                this is the point cloud

        Returns:
            persistence_pairs :
                this is an array of pairs (i,j). ``i``
                and ``j`` are the indices in the list ``phi`` associated to a
                positive-negative pair. When ``j`` is equal to ``-1``, it means
                that the match was not found and the feature
                is infinitely persistent. When ``(i,j)=(-1,-1)``, simply
                discard the pair
            phi :
                this is the 1d tensor of all filtration values
            homology_dims :
                this 1d tensor contains the homology
                group index
        """
        phi = self.phi(x)
        pairs, homology_dims = self._compute_pairs(x)
        approx_pairs: List[float] = list(np.around(pairs, self.approx_digits).flatten())
        num_approx = 10**self.approx_digits
        inv_num_approx = 10 ** (-self.approx_digits)
        phi_approx: Tensor = torch.round(phi * num_approx) * inv_num_approx
        # find the indices of phi_approx elements that are in approx_pairs
        indices_in_phi_of_pairs_ls: List = []
        for i in approx_pairs:
            indices_in_phi_of_pairs_ls += (
                torch.nonzero(phi_approx == i).flatten().detach().tolist()
            )
        indices_in_phi_of_pairs: Set = set(indices_in_phi_of_pairs_ls)
        persistence_pairs_array = -np.ones((len(approx_pairs), 2), dtype=np.int32)
        for i in indices_in_phi_of_pairs:
            try:
                temp_index: int = approx_pairs.index(
                    round(phi[i].item(), self.approx_digits)  # type: ignore
                )
                approx_pairs[temp_index] = float("inf")
                persistence_pairs_array[temp_index // 2, temp_index % 2] = int(i)
            except ValueError:
                pass
        return persistence_pairs_array, phi, homology_dims

    def _computing_persistence_with_gph(self, xx: Tensor) -> List:
        """This method accepts the pointcloud and returns the
        persistence diagram in the following form
        :math:`Pers:Filt_K \\subset \\mathbb R^{|K|} \\to (\\mathbb R^2)^p
        \\times \\mathbb R^q, \\Phi(X) \\mapsto D = \\cup_i^p
        (\\Phi_{\\sigma_i1}(X) ,\\Phi_{\\sigma_i2}(X) )
        \\times \\cup_j^q (\\Phi_{\\sigma_j}(X),+\\infty).`
        The persistence diagram can be readily used for
        gradient descent.

        Args:
            xx:
                point cloud with ``shape = (n_points, n_features)``

        Returns:
            list of shape (n, 3):
                Persistence pairs (corresponding to the
                first 2 dimensions) where the last dimension
                contains the homology dimension
        """
        output = ripser_parallel(
            xx.detach().numpy(),
            maxdim=max(self.homology_dimensions),
            thresh=self.max_edge_length,
            coeff=2,
            metric=self.metric,
            collapse_edges=self.collapse_edges,
            n_threads=-1,
            return_generators=True,
        )

        persistence_pairs = []

        for dim in self.homology_dimensions:
            if dim == 0:
                persistence_pairs += [
                    (0, torch.norm(xx[x[1]] - xx[x[2]]), 0) for x in output["gens"][dim]
                ]
            else:
                persistence_pairs += [
                    (
                        torch.norm(xx[x[1]] - xx[x[0]]),
                        torch.norm(xx[x[3]] - xx[x[2]]),
                        dim,
                    )
                    for x in output["gens"][1][dim - 1]
                ]
        return persistence_pairs

    def persistence_function(self, xx: Tensor) -> Tensor:
        """This is the Loss function to optimise.
        :math:`L=-\\sum_i^p |\\epsilon_{i2}-\\epsilon_{i1}|+
        \\lambda \\sum_{x \\in X} ||x||_2^2`
        It is composed of a regularisation term and a
        function on the filtration values that is (p,q)-permutation
        invariant.

        Args:
            xx:
                this is the persistence function argument, a tensor

        Returns:
            Tensor:
                the function value at ``xx``

        """
        out: Tensor = torch.tensor(0).to(torch.float)
        persistence_array: Union[List, Array]
        # this is much slower
        if self.directed and self.metric == "precomputed":
            persistence_array, phi, _ = self._persistence(xx)
            for item in persistence_array:  # type: ignore
                if item[1] != -1:
                    out += phi[item[1]] - phi[item[0]]
        else:
            persistence_array = self._computing_persistence_with_gph(xx)
            for item in persistence_array:
                out += item[1] - item[0]
        reg = (xx**2).sum()  # regularisation term
        return -out + self.zeta * reg  # maximise persistence

    def sgd(
        self, xx: Tensor, lr: float = 0.01, n_epochs: int = 5
    ) -> Tuple[Figure, Figure, List[float]]:  # type: ignore
        """This function is the core function of this class and uses the
        SGD method to move points around in order to optimise
        ``persistence_function``

        Args:
            xx:
                2d tensor representing the point cloud,
                the first dimension is ``n_points`` while the second
                ``n_features``
            lr:
                learning rate for the SGD
            n_epochs:
                the number of gradient iterations

        Returns:
           fig, fig3d, loss_val:
               respectively the plotly ``quiver_plot``, plotly ``cone_plot``
               ad the list of loss function values over the epochs"""

        if not type(xx) == Tensor:
            xx = torch.tensor(xx)
        xx.to(DEVICE)
        xx.requires_grad = True
        x: Array = np.array([])
        z: Array = np.array([])
        y: Array = np.array([])
        u: Array = np.array([])
        v: Array = np.array([])
        w: Array = np.array([])
        loss_val = []
        optimizer = optim.Adam([xx], lr=lr)
        for _ in tqdm(range(n_epochs)):
            optimizer.zero_grad()
            loss = self.persistence_function(xx)
            loss_val.append(loss.item())
            x = np.concatenate((x, xx.detach().numpy()[:, 0]))  # type: ignore
            y = np.concatenate((y, xx.detach().numpy()[:, 1]))  # type: ignore
            loss.backward()  # compute gradients and store them in Xp.grad
            grads = -xx.grad.detach()  # type: ignore
            u = np.concatenate((u, 1 / grads.norm(2, 1).mean() * grads.numpy()[:, 0]))  # type: ignore
            v = np.concatenate((v, 1 / grads.norm(2, 1).mean() * grads.numpy()[:, 1]))  # type: ignore
            try:
                z = np.concatenate((z, xx.detach().numpy()[:, 2]))  # type: ignore
                w = np.concatenate((w, 1 / grads.norm(2) * grads.numpy()[:, 2]))  # type: ignore
            except IndexError:
                z = np.concatenate((z, 0 * xx.detach().numpy()[:, 1]))  # type: ignore
                w = np.concatenate((w, 0 * grads.numpy()[:, 1]))  # type: ignore
            optimizer.step()
        fig = ff.create_quiver(x, y, u, v)
        fig3d = Figure(
            data=go.Cone(
                x=x,
                y=y,
                z=z,
                u=u,
                v=v,
                w=w,
                sizemode="absolute",
                sizeref=2,
                anchor="tip",
            )
        )
        return fig, fig3d, loss_val


# this function is outside the main class to use multiprocessing
def unpacking_apply_along_axis(tupl: List[Any]) -> Array:
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    func1d, axis, arr, args, kwargs = tupl
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def _combinations_with_single(tupl, max_length: int) -> Array:
    """Private function to compute combinations also for singletons
    with repetition. The un-intuitive shape of the output is
    becaue of the padding and vectorized operation happening
    in the next steps.

    Args:
        tupl :
            this is a list with only one element and such
            element is a tuple
        max_length :
            this is the paddding length.

    Returns:
        ndarray :
            padded list -- with (0,0) -- of all the combinations of indices
            within a given simplex."""
    tupl = tupl[0]  # the slice dimension will always contain 1 element
    if len(tupl) == 1:
        list1 = list(((tupl[0], tupl[0]),)) + list(
            (max_length - 1) * [(tupl[0], tupl[0])]
        )
        return np.array(list1)
    temp_list = list(combinations(tupl, 2))
    list2 = temp_list + ((max_length - len(temp_list)) * [(tupl[0], tupl[1])])
    return np.array(list2)


def comb(nnn: int, kkk: int) -> int:
    """this function computes :math:`\\frac{n!}{(k!*(n-k)!)}` efficiently"""
    kkk = min(kkk, nnn - kkk)
    numer = reduce(op.mul, range(nnn, nnn - kkk, -1), 1)
    denom = reduce(op.mul, range(1, kkk + 1), 1)
    return numer // denom
