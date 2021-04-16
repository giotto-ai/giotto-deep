import torch
from torch import nn
from torch import optim
from gtda.homology import VietorisRipsPersistence as vrp
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
from itertools import chain, combinations
import numpy as np
from tqdm import tqdm
import warnings
from math import comb
import multiprocessing

class PersistenceGradient():
    '''This clas computes the gradient of the persistence
    diagram with respect to a point cloud.
    
    Args:
        lr (float): learning rate for the SGD.
        n_epochs (int): the number of gradient iterations.
        Lambda (float): the relative weight of the regularisation part
            of the `persistence_function`
        homology_dimensions (tuple): tuple of homology dimensions
        collapse_edges (bool): whether to use Collapse or not
        max_edge_length (float or np.inf): the maximum edge length
            to be consider not infinity
        approx_digits (int): digits after which to trunc floats for
            list comparison
        metric (string): either `"euclidean"` or `"precomputed"`. The
            second oe is in case of X being the pairwise-distance matrix or
            the adjaceny matrix of a graph.
        
    '''
    def __init__(self,lr:float=0.001, n_epochs:int=10, Lambda:float = 0.5,
                 homology_dimensions=None,collapse_edges:bool=True, max_edge_length=np.inf,
                 approx_digits:int = 6, metric=None):

        self.collapse_edges = collapse_edges
        self.max_edge_length = max_edge_length
        if metric is None:
            self.metric = "euclidean"
        else:
            self.metric=metric
        self.approx_digits = approx_digits
        self.lr = lr
        self.n_epochs = n_epochs
        self.Lambda = Lambda
        if homology_dimensions is None:
            self.homology_dimensions = (0,1)
        else:
            self.homology_dimensions = homology_dimensions
        
    @staticmethod
    def powerset(iterable,max_length):
        '''powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        up to `max_length`.'''
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(0,max_length+1))
    
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
        n_processes = min(len(arr)//2,multiprocessing.cpu_count())
        #print(n_processes)
        # Chunks for the mapping (only a few chunks):
        chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
                  for sub_arr in np.array_split(arr, n_processes)]
        pool = multiprocessing.Pool(n_processes)
        individual_results = pool.map(unpacking_apply_along_axis, chunks)
        # Freeing the workers:
        pool.close()
        pool.join()

        return np.concatenate(individual_results)
        
    def _simplicial_pairs_of_indices(self,X):
        '''Private function to compute the pair of indice in X to
        matching the simplices.'''
        simplices = list(self.powerset(list(range(0,len(X))),max(self.homology_dimensions)+2))[1:]
        #print(len(simplices))
        simplices_array=np.array(simplices,dtype=object).reshape(-1,1)
        #print(simplices_array,type(simplices_array),simplices_array.shape)
        comb_number=comb(max(self.homology_dimensions)+2,2)
        #print("simplices computed")
        #the current computation bottleneck
        if len(simplices_array)>10000000:
            #print("parallel")
            pairs_of_indices = self._parallel_apply_along_axis(_combinations_with_single,1,simplices_array,comb_number)
        pairs_of_indices = np.apply_along_axis(_combinations_with_single,1,simplices_array,comb_number)
        #print("pairs of indices")
        return torch.tensor(pairs_of_indices)
    
    def Phi(self,X):
        '''This function is from $(R^d)^n$ to $R^{|K|}$,
        where K is the top simplicial complex of the VR filtration.
        It is ddefined as:
        $\Phi_{\sigma}(X)=max_{i,j \in \sigma}||x_i-x_j||.$ '''
        if self.metric=="precomputed":
            self.dist_mat = X
        else:
            self.dist_mat = torch.cdist(X,X)
        #print("done cdist ")
        simplicial_pairs = self._simplicial_pairs_of_indices(X).reshape(-1,2)
        #print("simplicial_pairs done")
        Is = simplicial_pairs[:,0]
        Js = simplicial_pairs[:,1]
        #print("compute phi")
        lista = torch.max((torch.gather(torch.index_select(self.dist_mat, 0, Is),1,Js.reshape(-1,1))).reshape(-1,comb(max(self.homology_dimensions)+2,2)),1)
        #print("phi is computed, but unsorted")
        #lista = [max([dist_mat[pair] for pair in pairs]) for pairs in simplicial_pairs if max([dist_mat[pair] for pair in pairs]) <= self.max_edge_length]
        lista = torch.sort(lista[0])[0] # not inplace
        return lista
        
    def _compute_pairs(self,X):
        '''Use giotto-tda to compute homology (b,d) pairs'''
        vr = vrp(metric=self.metric,homology_dimensions=self.homology_dimensions,
                 max_edge_length=self.max_edge_length,collapse_edges=self.collapse_edges)
        dgms = vr.fit_transform([X.detach().numpy()])
        pairs = dgms[0]#[:,:2]
        return pairs[:,:2], pairs[:,2]
        
    def _persistence(self,X):
        '''This function computess the persistence permutation.
        Thi permutation permutes the filtration values and matches
        them as follows:
        $Pers:Filt_K \subset \mathbb R^{|K|} \to (\mathbb R^2)^p \times \mathbb R^q, \Phi(X) \mapsto D = \cup_i^p (\Phi_{\sigma_i1}(X) , \Phi_{\sigma_i2}(X) ) \times \cup_j^q (\Phi_{\sigma_j}(X),+\infty).$
        
        Args:
            phi (list): this is the the list of all the ordered filtration values
            
        Returns:
            persistence_pairs (2darray): this is an array of pairs (i,j). `i` and `j` are the
                indices in the list `phi` associated to a positive-negative pair. When
                `j` is equal to `-1`, it means that the match was not found and the feature
                is infinitely persistent. When `(i,j)=(-1,-1)`, simply discard the pair.
        '''
        #print("computing phi")
        phi = self.Phi(X)
        pairs, homology_dims = self._compute_pairs(X)
        #print(pairs)
        #print("pairs computed")
        #phi = phi.detach().tolist()#torch.stack(phi).detach().tolist()
        #approx_pairs = [round(x,self.approx_digits) for x in list(chain(*pairs))]
        approx_pairs = list(np.around(pairs,self.approx_digits).flatten())
        num_approx =10**self.approx_digits
        inv_num_approx = 10**(-self.approx_digits)
        phi_approx = torch.round(phi*num_approx)*inv_num_approx
        #print("pairs approximated")
        # find the indices of phi_approx elements that are in the set approx_pairs
        indices_in_phi_of_pairs = []
        for i in approx_pairs:
            indices_in_phi_of_pairs+=torch.nonzero(phi_approx==i).flatten().detach().tolist()
        indices_in_phi_of_pairs=set(indices_in_phi_of_pairs)
        #print(indices_in_phi_of_pairs)
        #print(approx_pairs)
        #print(phi_approx)
        #indices_in_phi_of_pairs = [i for i in range(len(phi)) if round(phi[i],self.approx_digits) in approx_pairs]
        #print("indices found")
        persistence_pairs_array = -np.ones((len(indices_in_phi_of_pairs),2),dtype=np.int32)
        #print("start loop")
        for i in indices_in_phi_of_pairs:
            try:
                temp_index = approx_pairs.index(round(phi[i].item(),self.approx_digits))
                #print("added:",temp_index)
                approx_pairs[temp_index] = float('inf')
                persistence_pairs_array[temp_index//2,temp_index%2]=int(i)
            except:
                #print("not added:",i)
                pass
        #print("end loop")
        return persistence_pairs_array, phi

    def persistence_function(self,X):
        '''This is the Loss functon to optimise.
        $L=-\sum_i^p |\epsilon_{i2}-\epsilon_{i1}|+ \lambda \sum_{x in X} ||x||_2^2$
        It is composed of a regularisation term and a
        function on the filtration values that is (p,q)-permutation
        invariant.'''
        out = 0
        #print("run persistence")
        persistence_array, phi = self._persistence(X)
        for item in persistence_array:
            if item[1] != -1:
                out += (phi[item[1]]-phi[item[0]])
        reg = torch.trace(X.mm(X.T))
        return -out + self.Lambda*reg # maximise persistence and avoid points drifting away too much

    def SGD(self,X):
        '''This function is the core function of this class and uses the
        SGD method to move points around in ordder to optimise
        `persistence_function`.
        
        Args:
            Xp (torch.tensor): 2d tensor representing the point cloud,
                the first dimension is `n_points` while the second
                `n_features`.
        Returns:
           fig : plotly `quiver_plot`
           fig3d : plotly `cone_plot`
           loss_val (list): loss function values over the epochs'''
        if not type(X) == torch.Tensor:
            X = torch.tensor(X)
        X.requires_grad=True
        grads = torch.zeros_like(X)
        x = []
        z = []
        y = []
        u = []
        v = []
        w = []
        loss_val = []
        
        optimizer = optim.Adam([X], lr=self.lr)
        for _ in tqdm(range(self.n_epochs)):
            #print("start loop")
            optimizer.zero_grad()
            #print("create loss")
            loss = self.persistence_function(X)
            loss_val.append(loss.item())
            x = np.concatenate((x,X.detach().numpy()[:,0]))
            y = np.concatenate((y,X.detach().numpy()[:,1]))
            loss.backward() # compute gradients and store them in Xp.grad
            grads = -X.grad.detach()
            u = np.concatenate((u,1/grads.norm(2,1).mean()*grads.numpy()[:,0]))
            v = np.concatenate((v,1/grads.norm(2,1).mean()*grads.numpy()[:,1]))
            try:
                z = np.concatenate((z,X.detach().numpy()[:,2]))
                w = np.concatenate((w,1/grads.norm(2)*self.grads.numpy()[:,2]))
            except:
                z = np.concatenate((z,0*X.detach().numpy()[:,1]))
                w = np.concatenate((w,0*grads.numpy()[:,1]))
            optimizer.step()
        
        fig = ff.create_quiver(x,y,u,v)
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
def unpacking_apply_along_axis(tupl):
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
def _combinations_with_single(tupl,max_length):
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
    tupl = tupl[0] # the slice dimension will always contain one elemnt (a tuple)
    if len(tupl)==1:
        list1=list(((tupl[0],tupl[0]),))+list((max_length-1)*[(tupl[0],tupl[0])])
        return np.array(list1)
    temp_list = list(combinations(tupl,2))
    list2 = temp_list+((max_length-len(temp_list))*[(tupl[0],tupl[1])])
    return np.array(list2)
