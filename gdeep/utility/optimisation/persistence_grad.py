import torch
from torch import optim
#from gtda.homology import VietorisRipsPersistence as vrp
#from gtda.homology import FlagserPersistence as flp
from gph.python import ripser_parallel
from gtda.homology import FlagserPersistence as flp
import plotly.figure_factory as ff
import plotly.graph_objects as go
from itertools import chain, combinations
import numpy as np
from tqdm import tqdm
import multiprocessing
import operator as op
from functools import reduce
from typing import Iterator


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using GPU!")
else:
    DEVICE = torch.device("cpu")


class PersistenceGradient():
    '''This class computes the gradient of the persistence
    diagram with respect to a point cloud. The algorithms has
    first been developed in https://arxiv.org/abs/2010.08356 .

    Discalimer: this algorithm works well for generic point clouds.
    In case your point cloud has many simplices with same
    filtration values, the matching of the points to the persistent
    features may fail to disambiguate.

    Args:
        zeta (float): 
            the relative weight of the regularisation part
            of the `persistence_function`
        homology_dimensions (tuple): 
            tuple of homology dimensions
        collapse_edges (bool, default: False): 
            whether to use Collapse or not. Not implemented yet.
        max_edge_length (float or np.inf): 
            the maximum edge length
            to be consider not infinity
        approx_digits (int): 
            digits after which to trunc floats for
            list comparison
        metric (string): either `"euclidean"` or `"precomputed"`. 
            The second one is in case of X being 
            the pairwise-distance matrix or
            the adjaceny matrix of a graph.
        directed (bool): whether the input graph is a directed graph
            or not. Relevant only if `metric = "precomputed"`
        
    '''

    def __init__(self, zeta: float = 0.5, homology_dimensions: tuple = (0, 1),
                 collapse_edges: bool = False, max_edge_length: float = np.inf,
                 approx_digits: int = 6, metric: str = "euclidean",
                 directed: bool = False):

        self.collapse_edges = collapse_edges
        self.max_edge_length = max_edge_length
        self.metric = metric
        self.directed = directed
        self.approx_digits = approx_digits
        self.zeta = zeta
        self.homology_dimensions = homology_dimensions

    @staticmethod
    def powerset(iterable, max_length: int) -> Iterator:
        '''powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        up to `max_length`.'''
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for
                                   r in range(0, max_length + 1))

    @staticmethod
    def _parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
        """
        Like numpy.apply_along_axis(), but takes advantage of multiple
        cores.
        """
        # Effective axis where apply_along_axis() will be applied by each
        # worker (any non-zero axis number would work, so as to allow the use
        # of `np.array_split()`, which is only done on axis 0):
        effective_axis = 1 if axis == 0 else axis
        if effective_axis != axis:
            arr = arr.swapaxes(axis, effective_axis)
        n_processes = min(len(arr)//2, multiprocessing.cpu_count())
        # Chunks for the mapping (only a few chunks):
        chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
                  for sub_arr in np.array_split(arr, n_processes)]
        pool = multiprocessing.Pool(n_processes)
        individual_results = pool.map(unpacking_apply_along_axis, chunks)
        # Freeing the workers:
        pool.close()
        pool.join()

        return np.concatenate(individual_results)

    def _simplicial_pairs_of_indices(self, X):
        '''Private function to compute the pair of indices in X to
        matching the simplices.'''
        simplices = list(self.powerset(list(range(0, len(X))),
                         max(self.homology_dimensions) + 2))[1:]
        simplices_array = np.array(simplices, dtype=object).reshape(-1, 1)
        comb_number = comb(max(self.homology_dimensions)+2, 2)
        # the current computation bottleneck
        if len(simplices_array) > 10000000:
            pairs_of_indices = self._parallel_apply_along_axis(
                                    _combinations_with_single, 1,
                                    simplices_array, comb_number)
        else:
            pairs_of_indices = np.apply_along_axis(_combinations_with_single,
                                                   1,
                                                   simplices_array,
                                                   comb_number)
        return torch.tensor(pairs_of_indices, dtype=torch.int64)

    def phi(self, X: torch.Tensor) -> torch.Tensor:
        '''This function is from $(R^d)^n$ to $R^{|K|}$,
        where K is the top simplicial complex of the VR filtration.
        It is defined as:
        $\Phi_{\sigma}(X)=max_{i,j \in \sigma}||x_i-x_j||.$ '''

        if self.metric == "precomputed":
            self.dist_mat = X
        else:
            self.dist_mat = torch.cdist(X, X)
        simplicial_pairs = self._simplicial_pairs_of_indices(X).reshape(-1, 2)
        ks = simplicial_pairs[:, 0]
        js = simplicial_pairs[:, 1]
        comb_number = comb(max(self.homology_dimensions) + 2, 2)
        # morally, this is what one would like to do:
        # lista = [max([dist_mat[pair] for pair in pairs]) for pairs in
        #    simplicial_pairs if max([dist_mat[pair] for pair in pairs])
        #    <= self.max_edge_length]
        lista = torch.max((torch.gather(torch.index_select(self.dist_mat,
                                                           0, ks),
                          1, js.reshape(-1, 1))).reshape(-1,
                                                         comb_number), 1)[0]
        lista = torch.sort(lista)[0]  # not inplace
        return lista

    def _compute_pairs(self, X):
        '''Use giotto-tda to compute homology (b,d) pairs

        Args:
            X (tensor): this is the input point cloud or the input
                pairwise distance or the adjacency matrix of a graph

        Returns:
            pairs (tensor): this tensor is the tensor obtained from
                the list of pairs (b,d)
            homology_dim (tensor): this 1d tensor contains the homology
                group index'''

        dgms = flp().fit_transform([X.detach().numpy()])
        pairs = dgms[0]
        return pairs[:, :2], pairs[:, 2]

    def _persistence(self, X):
        '''This function computes the persistence permutation.
        This permutation permutes the filtration values and matches
        them as follows:
        $Pers:Filt_K \subset \mathbb R^{|K|} \to (\mathbb R^2)^p
        \times \mathbb R^q, \Phi(X) \mapsto D = \cup_i^p
        (\Phi_{\sigma_i1}(X) ,\Phi_{\sigma_i2}(X) )
        \times \cup_j^q (\Phi_{\sigma_j}(X),+\infty).$

        Args:
            X (np.array): this is the point cloud

        Returns:
            persistence_pairs (2darray): this is an array of pairs (i,j). `i`
                and `j` are the indices in the list `phi` associated to a
                positive-negative pair. When `j` is equal to `-1`, it means
                that the match was not found and the feature
                is infinitely persistent. When `(i,j)=(-1,-1)`, simply
                discard the pair
            phi (tensor): this is the 1d tensor of all filtration values
            homology_dims (tensor): this 1d tensor contains the homology
                group index
        '''
        phi = self.phi(X)
        pairs, homology_dims = self._compute_pairs(X)
        approx_pairs = list(np.around(pairs, self.approx_digits).flatten())
        num_approx = 10**self.approx_digits
        inv_num_approx = 10**(-self.approx_digits)
        phi_approx = torch.round(phi*num_approx)*inv_num_approx
        # find the indices of phi_approx elements that are in approx_pairs
        indices_in_phi_of_pairs = []
        for i in approx_pairs:
            indices_in_phi_of_pairs += \
                torch.nonzero(phi_approx == i).flatten().detach().tolist()
        indices_in_phi_of_pairs = set(indices_in_phi_of_pairs)
        persistence_pairs_array = -np.ones((len(approx_pairs), 2),
                                           dtype=np.int32)
        for i in indices_in_phi_of_pairs:
            try:
                temp_index = approx_pairs.index(round(phi[i].item(),
                                                self.approx_digits))
                approx_pairs[temp_index] = float('inf')
                persistence_pairs_array[temp_index//2,
                                        temp_index % 2] = int(i)
            except ValueError:
                pass
        return persistence_pairs_array, phi, homology_dims
    
    def _computing_persistence_with_gph(self, X: torch.Tensor) -> list:
        """This method accepts the pointcloud and returns the
        persistence diagram in the following form
        $Pers:Filt_K \subset \mathbb R^{|K|} \to (\mathbb R^2)^p
        \times \mathbb R^q, \Phi(X) \mapsto D = \cup_i^p
        (\Phi_{\sigma_i1}(X) ,\Phi_{\sigma_i2}(X) )
        \times \cup_j^q (\Phi_{\sigma_j}(X),+\infty).$
        The persstence diagram ca be readily used for 
        gradient descent.

        Args:
            X (torch.tensor):
                point cloud

        Returns:
            list of shape (n, 3):
                Persistence pairs (correspondig to the
                first 2 dimensions) where the last dimension 
                contains the homology dimension
        """
        output = ripser_parallel(X.detach().numpy(),
                                 maxdim=max(self.homology_dimensions),
                                 thresh=self.max_edge_length,
                                 coeff=2,
                                 metric=self.metric,
                                 collapse_edges=self.collapse_edges,
                                 n_threads=-1,
                                 return_generators=True)
        
        persistence_pairs = []
        #print(output["gens"])
        for dim in self.homology_dimensions:
            if dim == 0:
                persistence_pairs += [(0, torch.norm(X[x[1]]-X[x[2]]),
                                      0) for x in output["gens"][dim]]
            else:
                persistence_pairs += [(torch.norm(X[x[1]]-X[x[0]]), 
                                      torch.norm(X[x[3]]-X[x[2]]), 
                                      dim) for x in output["gens"][1][dim-1]]
        return persistence_pairs
        

    def persistence_function(self, X: torch.Tensor) -> torch.Tensor:
        '''This is the Loss functon to optimise.
        $L=-\sum_i^p |\epsilon_{i2}-\epsilon_{i1}|+
        \lambda \sum_{x in X} ||x||_2^2$
        It is composed of a regularisation term and a
        function on the filtration values that is (p,q)-permutation
        invariant.'''
        out = 0
        # this is much slower
        if self.directed and self.metric == "precomputed":
            persistence_array, phi, _ = self._persistence(X)
            for item in persistence_array:
                if item[1] != -1:
                    out += (phi[item[1]]-phi[item[0]])
        else:
            persistence_array = self._computing_persistence_with_gph(X)
            for item in persistence_array:
                out += item[1]-item[0]
        reg = (X**2).sum()  # regularisation term
        return -out + self.zeta*reg  # maximise persistence

    def SGD(self, X: torch.Tensor, lr: float = 0.01, n_epochs: int = 5):
        '''This function is the core function of this class and uses the
        SGD method to move points around in ordder to optimise
        `persistence_function`

        Args:
            Xp (torch.tensor): 2d tensor representing the point cloud,
                the first dimension is `n_points` while the second
                `n_features`
            lr (float): learning rate for the SGD
            n_epochs (int): the number of gradient iterations

        Returns:
           fig : plotly `quiver_plot`
           fig3d : plotly `cone_plot`
           loss_val (list): loss function values over the epochs'''

        if not type(X) == torch.Tensor:
            X = torch.tensor(X)
        X.to(DEVICE)
        X.requires_grad = True
        grads = torch.zeros_like(X)
        x = np.array([])
        z = np.array([])
        y = np.array([])
        u = np.array([])
        v = np.array([])
        w = np.array([])
        loss_val = []
        optimizer = optim.Adam([X], lr=lr)
        for _ in tqdm(range(n_epochs)):
            optimizer.zero_grad()
            loss = self.persistence_function(X)
            loss_val.append(loss.item())
            x = np.concatenate((x, X.detach().numpy()[:, 0]))
            y = np.concatenate((y, X.detach().numpy()[:, 1]))
            loss.backward()  # compute gradients and store them in Xp.grad
            grads = -X.grad.detach()
            u = np.concatenate((u, 1/grads.norm(2,
                                1).mean()*grads.numpy()[:, 0]))
            v = np.concatenate((v, 1/grads.norm(2,
                                1).mean()*grads.numpy()[:, 1]))
            try:
                z = np.concatenate((z, X.detach().numpy()[:, 2]))
                w = np.concatenate((w,
                                    1/grads.norm(2) *
                                    grads.numpy()[:, 2]))
            except IndexError:
                z = np.concatenate((z, 0*X.detach().numpy()[:, 1]))
                w = np.concatenate((w, 0*grads.numpy()[:, 1]))
            optimizer.step()
        fig = ff.create_quiver(x, y, u, v)
        fig3d = go.Figure(data=go.Cone(
            x=x,
            y=y,
            z=z,
            u=u,
            v=v,
            w=w,
            sizemode="absolute",
            sizeref=2,
            anchor="tip"))
        return fig, fig3d, loss_val


# this function is outside the main class to use multiprocessing
def unpacking_apply_along_axis(tupl: tuple) -> np.ndarray:
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


# this function is outside the main class to use multiprocessing
def _combinations_with_single(tupl, max_length):
    '''Private function to compute combinations also for singletons
    with repetition. The un-intuitive shape of the output is
    becaue of the padding and vectorized operation happening
    in the next steps.

    Args:
        tupl (list): this is a list with only one element and such
            element is a tuple
        max_length (int): this is the paddding length.

    Returns:
        padded list -- with (0,0) -- of all the combinations of indices
            within a given simplex.'''
    tupl = tupl[0]  # the slice dimension will always contain 1 element
    if len(tupl) == 1:
        list1 = list(((tupl[0], tupl[0]), )) + \
            list((max_length-1)*[(tupl[0], tupl[0])])
        return np.array(list1)
    temp_list = list(combinations(tupl, 2))
    list2 = temp_list+((max_length-len(temp_list))*[(tupl[0], tupl[1])])
    return np.array(list2)


def comb(nnn: int, kkk: int) -> int:
    '''this function computes n!/(k!*(n-k)!) efficiently'''
    kkk = min(kkk, nnn-kkk)
    numer = reduce(op.mul, range(nnn, nnn-kkk, -1), 1)
    denom = reduce(op.mul, range(1, kkk+1), 1)
    return numer // denom
