import numpy as np
import plotly.express as px
import networkx as nx
from networkx.algorithms.clique import find_cliques, enumerate_all_cliques
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from networkx.drawing.nx_pylab import draw_kamada_kawai
import warnings
from networkx.algorithms.components import number_connected_components
from itertools import combinations, chain
from sklearn.base import BaseEstimator, TransformerMixin

class IntersectionHomology(BaseEstimator, TransformerMixin):
    '''This class computes the Intersection Homology of a clque complex.
    After initialising the class, a stratification is computed (during `fit`).
    Once the stratification is established, the Intersection homology
    can be computed (during `transform`).
    
    Args:
        -
    
    '''
    def __init__(self,knn:int=5,perversity=None, perversity_check=True,coarse_strata=False):
        self.knn = knn
        self.gg = None
        if perversity is None:
            self.perversity=self.p
        else:
            self.perversity = perversity
        self.check_is_fitted=False
        self.perversity_check = perversity_check
        self.list_proper = None
        self.coarse_strata = coarse_strata
    
    
    def fit(self,X):
        ''' The fit functions accepts as input a pointcloud as a
        2darray or a networkx graph.
        
        Args:
            - X 2darray OR networkx Graph object:
            this is the inout simplicial complex or the point cloud
            from which to compute the knn graph
        
        '''
        self.check_is_fitted = True
        if type(X)==np.ndarray or type(X)==list:
            self.gg = self.from_pts_to_nn_graph(X,self.knn)
        else:
            self.gg = X
        self.simplices, self.maximal_cliques = self.from_graph_to_complex(self.gg)
        if self.perversity_check:
            self.list_proper = self.perv_list(self.perversity)
        return self
    
    def transform(self,X,phi_list_proper=None):
        if self.check_is_fitted:
            if phi_list_proper is None:
                if self.perversity_check:
                    return self.phi_generators(self.simplices,self.list_proper)
                return self.generators(self.simplices)
            else:
                return self.phi_generators(self.simplices,phi_list_proper)
        warnings.warn("You need to fit before tranforming!")
        
    def plot(self):
        if self.check_is_fitted:
            draw_kamada_kawai(self.gg,with_labels=True)
    
    @staticmethod
    def _reduce(D1,verbose:bool = False):
        '''Input: incidence matrix (2d array)
        Output: same dimensioal matrix but reduced'''
        V_fin = np.eye(D1.shape[1])
        V_temp = np.eye(D1.shape[1])
        for j in range(D1.shape[1]):
            j1=0
            while j1 < j:
                #print(j,j1)
                if IntersectionHomology._low_(D1,j1) == IntersectionHomology._low_(D1,j) and IntersectionHomology._low_(D1,j) != 0:
                    
                    D1.T[j] = (D1.T[j] + D1.T[j1])%2
                    V_temp[j1,j] = 1
                    V_fin = V_fin.dot(V_temp).copy()%2
                    V_temp[j1,j] = 0
                    if verbose:
                        print("summing line ",j1," to ",j)
                        px.imshow(D1).show()
                    j1=-1
                j1+=1
        return D1, V_fin
    
    def _compute_homology(self, D1, proper_list = None):
        '''returns the index of the unpaired simplices. i.e. those
        that generate the homology group.'''
        threshold = D1.shape[0]
        if proper_list is not None:
            threshold = len(proper_list)
        # simplicial pairs
        D1, _ = self._reduce(D1)
        index_positive_simplices = []
        index_pairings = []
        for j in range(D1.shape[1]):
            i=self._low_(D1,j)
            if i == 0:
                index_positive_simplices.append(j)
            else:
                if i < threshold:
                    index_pairings.append([i,j])

        # check if positive is in pairings
        homology_index = []
        for index in index_positive_simplices:
            if not np.array([True if index in pair else False for pair in index_pairings]).any():
                homology_index.append(index)
        if proper_list is not None:
            return np.array(proper_list)[homology_index]
        return homology_index

    @staticmethod
    def _low_(M,j):
        column = M.T[j]
        try:
            return np.max(np.nonzero(column))
        except:
            return 0

    def phi(self,D,lista:list):
        '''inputs the incidence matrix and outputs the m x s incidence matrix,
        with the rows ordered (first proper, then improper)'''
        improper = list(set(range(D.shape[1]))-set(lista))
        proper = lista
        #print(improper, proper)
        return ((D.T[proper]).T)[proper + improper], proper


    def from_pts_to_nn_graph(self,X,knn):
        '''get the knn graph of the point cloud'''
        A = kneighbors_graph(X, knn, include_self=False)
        Mat = A.toarray()
        G = nx.convert_matrix.from_numpy_matrix(Mat)
        return G

    def from_graph_to_complex(self,G):
        '''Get the list of simplices (cliques) fro
        the input graph.
        Input: networkx graph G
        Output: (list of simplices , list of maximal_simplices)'''
        simplices = list(enumerate_all_cliques(G))
        max_cliques=list(map(list,list(map(set,list(find_cliques(G))))))
        map(list.sort,max_cliques)
        #simplices_dupl = list(map(list,np.array(G.nodes).reshape(-1,1))) + list(map(list,list(G.edges))) + cliques
        #simplices = []
        #[list.sort(s) for s in simplices_dupl]
        #[simplices.append(n) for n in simplices_dupl if n not in simplices]
        #simplices.sort(key=len)
        return simplices, max_cliques
    
    @staticmethod
    def is_sublist(sub_list, test_list):
        if(all(x in test_list for x in sub_list)):
            return True
        return False

    def _boundary_matrix(self,simplices):
        '''build the incidence matrix from the list of simplices.
        If there are `n` simplices, the matrx is `n * n`. '''
        row = []
        col = []
        data = []
        length = len(simplices)
        for i in range(length):
            for j in range(length):
                #print(i,j)
                #print(simplices[i],simplices[j])
                if len(simplices[i]) == len(simplices[j])-1:
                    if self.is_sublist(simplices[i], simplices[j]):
                        row.append(i)
                        col.append(j)
                        data.append(1)
                        #print("added")
                #if len(simplices[i]) == len(simplices[j])+1:
                #    if is_sublist(simplices[j], simplices[i]):
                #        row.append(i)
                #        col.append(j)
                #        data.append(1)

        boundary_matrix = csr_matrix((data, (row, col)),shape=(length,length)).toarray()
        return boundary_matrix

    def homology_array(self,simplices):
        '''outputs the ordered list of betti numbers,
        e.g. [0,0,0,1,1,2,2,3] for b_0=3, b_1=2, b_2=2, b_3=1'''
        bdry_mat = self._boundary_matrix(simplices)
        comp_hom = self._compute_homology(bdry_mat)
        homology_generators = np.array(list(map(len,simplices)))[comp_hom]-1
        return homology_generators, self._reduce(bdry_mat)[1].T[comp_hom]

    def phi_homology_array(self,simplices,list_proper):
        '''outputs the ordered list of phi-betti numbers,
        e.g. [0,0,0,1,1,2,2,3] for b_0=3, b_1=2, b_2=2, b_3=1'''
        D = self._boundary_matrix(simplices)
        comp_hom = self._compute_homology(*self.phi(D,list_proper))
        homology_generators = np.array(list(map(len,simplices)))[comp_hom]-1
        return homology_generators, self._reduce(D)[1].T[comp_hom]

    def generators(self,simplices):
        '''Input smplices and prints the genetarors and the
        homology degree.'''
        arr, vv = self.homology_array(simplices)
        output = []
        for index in range(len(arr)):
            output.append((arr[index], list(np.array(simplices,dtype=object)[(np.nonzero(vv[index])[0])])))
        return output

    def phi_generators(self,simplices,list_proper):
        '''Input smplices and the index list of proper ones,
        Output: prints the genetarors and the
        homology degree.'''
        arr, vv = self.phi_homology_array(simplices,list_proper)
        output = []
        for index in range(len(arr)):
            output.append((arr[index], list(np.array(simplices,dtype=object)[(np.nonzero(vv[index])[0])])))
        return output

    # top perversity
    @staticmethod
    def p(k):
        if k<1:
            return 0
        return k-2

    # zero perversity
    @staticmethod
    def q(k):
        return 0

    def check_stratum(self,sigma,maximal_cliques):
        '''define a stratum to be a set of simplices such that each simplex in one stratum is:
            1. it is a maximal d-dimensional clique
            2. OR it is a d-dimensional simplex such that it belogs to 3 or more max simplices that
            are not otherwise connected. If they are otherwise connected, the graph of cthe max
            simplices connectivity is computed and analysed. If there are more than one connected components,
            then we have a d-dimensional piece of a stratum
            3. OR it belongs to exactly 2 max cliques of dimension larger than d+2
            '''
        cond1 = sigma in maximal_cliques
        num_inter = np.sum([set(sigma).issubset(set(max_spx)) for max_spx in maximal_cliques])
        cond2 = False
        if num_inter>2:
            intersecting_max_spxs=[max_spx for max_spx in maximal_cliques if set(sigma).issubset(set(max_spx)) ]
            int_num = 0
            G_new = nx.Graph()
            for s in intersecting_max_spxs:
                #print("intersection max simplices : ", intersecting_max_spxs)
                #print("removing s : ", list(filter(lambda x: x != s, intersecting_max_spxs)))
                #proto_edges = np.nonzero([len(set(s).intersection(ss))>len(sigma) for ss in list(filter(lambda x: x != s, intersecting_max_spxs))])
                proto_edges=np.nonzero([len(set(s).intersection(ss))>len(sigma) for ss in intersecting_max_spxs])[0]
                #print("proto_edges", proto_edges)
                edges_of_G = list(combinations(proto_edges, 2))
                #print("edges_of_G", edges_of_G)
                G_new.add_edges_from(edges_of_G)
            num = number_connected_components(G_new)
            cond2 = (num > 1)
        cond3 = False
        if num_inter==2:
            cond3 = np.sum([set(sigma).issubset(set(max_spx))*(len(max_spx)-len(sigma))>1 for max_spx in maximal_cliques])==2
        return cond1 or cond2 or cond3



    #make sure that with q, all is as in the standard case
    def perv_list(self, perversity):
        '''This function produces the list of proper simplices, to be
        inputted directly to the `fit` method.
        '''
        n=max(map(len,self.simplices))
        # there is the option to get the coarse stratification with the next line. also decomment the stratum below
        if self.coarse_strata:
            strata = compute_coarsest_strata(self.simplices)
        perverse_list = []
        for i, sigma in enumerate(self.simplices):
            check = 0
            for k in range(n):
                
                stratum = [s for s in self.simplices if len(s) == n-k and self.check_stratum(s,self.maximal_cliques)]#to be double checked
                #print(stratum, sigma)
                if self.coarse_strata:
                    stratum = strata[n-k-1]
                else:
                    stratum = [s for s in self.simplices if len(s) == n-k and self.check_stratum(s,self.maximal_cliques)]#to be double checked
                #print(stratum, sigma)
                if len(stratum)>0:
                    intersection_temp = [set(sigma).intersection(set(simplex_stratum)) for simplex_stratum in stratum]
                    top_dim = max(map(len,intersection_temp))-1
                    if top_dim == -1:
                        top_dim = - np.inf
                    #print("top dim: ", top_dim)
                    if top_dim <= len(sigma)-1-k+perversity(k):
                        check+=1
                        #print("ok, ", k)
                else:
                    #print("ok, ", k)
                    check+=1
            if check == n:
                #print("appending ",sigma)
                perverse_list.append(i)
                #list(range(len(gg.nodes))) +
        return perverse_list


def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(map(list,chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1,1))))

def _star(s, simplices):
    return [x for x in simplices if s in x]

def _intersect_lists(ls1,ls2):
    '''intersect lists of lists'''
    return [s for s in ls1 if s in ls2]

def compute_coarsest_strata(simplices):
    d = max(map(len, simplices))
    X_d = [s for s in simplices if len(s)<=d]
    Sj = [[] for k in range(d)]
    while d>=1:
        for x in [s for s in X_d if len(s)==d]:
            Bx = _star(x,simplices)
            #print("Bx:",Bx)
            set_to_check = _intersect_lists(Bx,X_d)
            #print("stc:", set_to_check)
            for w in set_to_check:
                for y in set_to_check:
                    By = _star(y,simplices)
                    Bw = _star(w,simplices)
                    #y>=w i.e. By <= Bw
                    if IntersectionHomology.is_sublist(By,Bw):
                        Sj[d-1].append(x)
            if len(Bx)==0:
                Sj[d-1].append(x)
        #print("Sj:",Sj)
        Sj_cpx = list(chain.from_iterable(map(_powerset,Sj[d-1])))
        #print("cpx:",Sj_cpx)
        lista = [s for s in X_d if s not in Sj_cpx]
        #print("lista:",lista)
        if len(lista)==0:
            d-=1
            X_d = [s for s in X_d if len(s)<=d and s not in Sj_cpx]
        else:
            d = max(map(len, lista))
            X_d = [s for s in lista if len(s)<=d and s not in Sj_cpx]
        #print("X_d:",X_d)
    return Sj
