import torch
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

class PersistenceGradient():
    '''This clas computes the gradient of the persistence
    diagram with respect to a point cloud.
    
    Args:
        Xp (torch.tensor): 2d tensor representing the point cloud,
            the first dimension is `n_points` while the second
            `n_features`.
        lr (float): learning rate for the SGD.
        n_epochs (int): the number of gradient iterations.
        Lambda (float): the relative weight of the regularisation part
            of the `persistence_function`.
        homology_dimensions (tuple): tuple of homology dimensions.
        approx_digits (int): digits after which to trunc floats
        
    '''
    def __init__(self,Xp,lr:float=0.001, n_epochs:int=10, Lambda:float = 0.5,
                 homology_dimensions=None,collapse_edges=True, max_edge_length=np.inf,
                 approx_digits:int = 7):
        self.Xp = Xp.detach().clone()
        self.Xp.requires_grad=True
        self.collapse_edges = collapse_edges
        self.max_edge_length = max_edge_length
        if self.Xp.shape[0]==self.Xp.shape[1]:
            self.metric="precomputed"
        else:
            self.metric="euclidean"
        self.approx_digits = approx_digits
        self.lr = lr
        self.n_epochs = n_epochs
        self.Lambda = Lambda
        self.grads = torch.zeros_like(self.Xp)
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
    def _combinations_with_single(tupl):
        '''Private function to compute combinations also for singletons
        with repetition.'''
        if len(tupl)==1:
            return ((tupl[0],tupl[0]),)
        return tuple(combinations(tupl,2))
        
    def _simplicial_pairs_of_indices(self,X):
        '''Private function to compute the pair of indice in X to
        matching the iimplices.'''
        simplices = list(self.powerset(list(range(0,len(X))),max(self.homology_dimensions)+2))[1:]
        #print("len simplx",len(simplices))
        pairs_of_indices = [self._combinations_with_single(s) for s in simplices]
        return pairs_of_indices
    
    def Phi(self,X):
        '''This function is from $(R^d)^n$ to R^{|K|},
        where K is the top simplicial complex of the VR filtration.
        It is ddefined as:
        $\Phi_{\sigma}(X)=max_{i,j \in \sigma}||x_i-x_j||.$ '''
        if self.metric=="precomputed":
            dist_mat = X
            
        else:
            dist_mat = torch.cdist(X,X)
        lista = [max([dist_mat[pair] for pair in pairs]) for pairs in self._simplicial_pairs_of_indices(X)]
        lista.sort() # inplace
        return lista
        
    def _compute_pairs(self):
        '''Use giotto-tda to compute homology (b,d) pairs'''
        vr = vrp(metric=self.metric,homology_dimensions=self.homology_dimensions,
                 max_edge_length=self.max_edge_length,collapse_edges=self.collapse_edges)
        dgms = vr.fit_transform([self.Xp.detach().numpy()])
        pairs = dgms[0]#[:,:2]
        return pairs[:,:2], pairs[:,2]
        
    def _persistence(self,phi):
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
        pairs, homology_dims = self._compute_pairs()
        #print("pairs computed")
        phi = torch.stack(phi).detach().tolist()
        #print("got phi")
        approx_pairs = [round(x,self.approx_digits) for x in list(chain(*pairs))]
        #print("pairs approximated")
        indices_in_phi_of_pairs = [i for i in range(len(phi)) if round(phi[i],self.approx_digits) in approx_pairs]
        #print("indices found")
        persistence_pairs_array = -np.ones((len(indices_in_phi_of_pairs),2),dtype=np.int32)
        #print("start loop")
        for i in indices_in_phi_of_pairs:
            try:
                temp_index = approx_pairs.index(round(phi[i],self.approx_digits))
                approx_pairs[temp_index] = np.inf
                persistence_pairs_array[temp_index//2,temp_index%2]=int(i)
            except:
                pass
        return persistence_pairs_array

    def persistence_function(self,X):
        '''This is the Loss functon to optimise.
        $L=-\sum_i^p |\epsilon_{i2}-\epsilon_{i1}|+ \lambda \sum_{x in X} ||x||_2^2$
        It is composed of a regularisation term and a
        function on the filtration values that is (p,q)-permutation
        invariant.'''
        out = 0
        phi = self.Phi(X)
        persistence_array = self._persistence(phi)
        for item in persistence_array:
            if item[1] != -1:
                out += (phi[item[1]]-phi[item[0]])
        reg = torch.trace(X.mm(X.T))
        return -out + self.Lambda*reg # maximise persistence and avoid points drifting away too much

    def plot(self):
        '''plot the vector field, in 2D, of the point cloud with its gradients'''
        x,y,u,v = self.Xp.detach().numpy()[:,0], self.Xp.detach().numpy()[:,1], 10*self.lr*self.grads.numpy()[:,0], 10*self.lr*self.grads.numpy()[:,1]
        try:
            z = self.Xp.detach().numpy()[:,2]
            w = 10*self.lr*self.grads.numpy()[:,2]
            fig = go.Figure(data=go.Cone(
                x=x,
                y=y,
                z=z,
                u=u,
                v=v,
                w=w,
                sizemode="absolute",
                sizeref=2,
                anchor="tip"))
        except:
            fig = ff.create_quiver(x,y,u,v)
        
        fig.show()

    def SGD(self):
        '''This function is the core function of this class and uses the
        SGD method to move points around in ordder to optimise
        `persistence_function`.
        
        Returns:
           fig : plotly `quiver_plot`
           fig3d : plotly `cone_plot`
           loss_val (list): loss function values over the epochs'''
        x = []
        z = []
        y = []
        u = []
        v = []
        w = []
        loss_val = []
        
        optimizer = optim.Adam([self.Xp], lr=self.lr)
        for _ in tqdm(range(self.n_epochs)):
            optimizer.zero_grad()
            loss = self.persistence_function(self.Xp)
            loss_val.append(loss.item())
            x = np.concatenate((x,self.Xp.detach().numpy()[:,0]))
            y = np.concatenate((y,self.Xp.detach().numpy()[:,1]))
            loss.backward() # compute gradients and store them in Xp.grad
            self.grads = -self.Xp.grad.detach()
            u = np.concatenate((u,10*self.lr*self.grads.numpy()[:,0]))
            v = np.concatenate((v,10*self.lr*self.grads.numpy()[:,1]))
            try:
                z = np.concatenate((z,self.Xp.detach().numpy()[:,2]))
                w = np.concatenate((w,10*self.lr*self.grads.numpy()[:,2]))
            except:
                z = np.concatenate((z,0*self.Xp.detach().numpy()[:,1]))
                w = np.concatenate((w,0*self.grads.numpy()[:,1]))
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
