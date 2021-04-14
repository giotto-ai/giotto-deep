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
        
    '''
    def __init__(self,Xp,lr:float=0.01, n_epochs:int=10, Lambda:float = 0.5,
                 homology_dimensions=None):
        self.Xp = Xp.detach().clone()
        self.Xp.requires_grad=True
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
        dist_mat = torch.cdist(X,X)
        lista = [max([dist_mat[pair] for pair in pairs]) for pairs in self._simplicial_pairs_of_indices(X)]
        lista.sort() # inplace
        return lista
        
    def _compute_pairs(self):
        '''Use giotto-tda to compute homology (b,d) pairs'''
        vr = vrp(homology_dimensions=self.homology_dimensions)
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
            persistence_pairs (list): this is a list of triplets (i,j,homology_dim). `i` and `j` are the
                indices in the list `phi` associated to a positive-negative pair.
        '''
        num_tolerance = 1e-04
        pairs, homology_dims = self._compute_pairs()
        #print(pairs)
        #print(homology_dims)
        phi = torch.stack(phi)
        #print(phi.shape)
        phi_to_remove = []
        persistence_pairs = []
        persistence_pairs_set = set()
        for i,phi_i in enumerate(phi):
            # check if phi_i is in `pairs`
                # if so, match it with the other element of the pair
                # add i,j and j,i elements in the `persistence_pairs`
                # drop the pair from pairs and from `phi_to_remove`
            # if not, check if there is another equal element in `phi_to_remove`
                # if so, add the i,j and j,i elements in the `persistence_pairs`
                # drop from `phi_to_remove`
                # if not, then infinite persistence
            if i not in persistence_pairs_set:
                #print("phi_i:",phi_i)
                try:
                    index=torch.nonzero(torch.isclose(torch.from_numpy(pairs).float(),phi_i,rtol=num_tolerance))
                    if len(index)>0:
                        #print("index:",index)
                        equal_number=pairs[index[0,0]][(index[0,1]+1)%2]
                        #print("paired filtr value: ", equal_number)
                        j = torch.argmin(torch.abs(phi-torch.tensor(equal_number,dtype=torch.float32)))
                        if j.item() not in persistence_pairs_set:
                            persistence_pairs.append((i,j.item(),homology_dims[index[0,0]]))
                            #print(homology_dims[index[0,0]])
                            persistence_pairs_set.add(i)
                            persistence_pairs_set.add(j.item())
                        else:
                            persistence_pairs.append((i,np.inf,homology_dims[index[0,0]]))
                            #print(homology_dims[index[0,0]])
                            persistence_pairs_set.add(i)
                        #print("pair:",i,j)
                        pairs = np.vstack((pairs[:index[0,0]],pairs[index[0,0]+1:]))
                        homology_dims = np.concatenate((homology_dims[:index[0,0]],homology_dims[index[0,0]+1:]))
                        #print(homology_dims)
                        phi_to_remove.append(phi_i)
                        phi_to_remove.append(phi[j])
                        #print("remaining pairs:",pairs)
                        #print("remaining phi:", phi_to_remove)
                    else:
                        at_least_two=torch.nonzero(torch.isclose(phi,phi_i,atol=num_tolerance))
                        #print("at least two:", at_least_two)
                        if len(at_least_two)>1:
                            j = at_least_two[1,0]
                            #print("pair:",i,j)
                            if j.item() not in persistence_pairs_set:
                                persistence_pairs.append((i,j.item(),0.)) # zero as default
                                persistence_pairs_set.add(i)
                                persistence_pairs_set.add(j.item())
                            else:
                                persistence_pairs.append((i,np.inf,0.)) # zero as default
                                persistence_pairs_set.add(i)
                            phi_to_remove.append(phi[at_least_two[0,0]])
                            phi_to_remove.append(phi[at_least_two[1,0]])
                            #print("remaining phi:", phi_to_remove)
                        else:
                            index=torch.nonzero(torch.isclose(torch.stack(phi_to_remove),phi_i,rtol=num_tolerance))
                            if len(index)==0:
                                persistence_pairs.append((i,np.inf,0.)) # zero as default
                                persistence_pairs_set.add(i)
                except:
                    warnings.warn("Numerical approximation errors! Incresing tolerance.")
                    num_tolerance = 10*num_tolerance
                    
        #print(persistence_pairs)
        return persistence_pairs

    def persistence_function(self,X):
        '''This is the Loss functon to optimise.
        It is composed of a regularisation term and a
        funtion on the filtration values that is (p,q)-permutation
        invariant.'''
        out = 0
        phi = self.Phi(X)
        persistence_array = self._persistence(phi)
        for item in persistence_array:
            if item[1] is not np.inf:
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
