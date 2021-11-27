import numpy as np
from sklearn.metrics import pairwise_distances
from gtda.homology import VietorisRipsPersistence
from torch.utils.data import DataLoader, TensorDataset  # type: ignore

from joblib import Parallel, delayed

class CurvatureSamplingGenerator(object):
    
    def __init__(self,
                 curvature_interval=(-2.0, 2.0),
                 num_samplings=100,
                 num_points_per_sampling=1_000,
                 homology_dimensions=(0, 1),
                 n_jobs=1,
    ):
        assert len(curvature_interval) == 2, "Curvature interval must be 2-dimensional"
        self._curvature_interval = curvature_interval
        self._num_samplings = num_samplings
        self._num_points_per_sampling = num_points_per_sampling
        self._homology_dimensions = homology_dimensions
        
        self._curvatures = np.random.uniform(low=curvature_interval[0],
                                             high=curvature_interval[1],
                                             size=(self._num_samplings))
        
        self._n_jobs = n_jobs
        
        self._compute_diagrams()
        
    
    def _phi(self, curvature, u_vect):
        if curvature > 0:
            r = (2/np.sqrt(curvature)) * np.arcsin(np.sqrt(u_vect) * np.sin(np.sqrt(curvature)/2))
        if curvature == 0:
            r = np.sqrt(u_vect)
        if curvature < 0:
            r = (2/np.sqrt(-curvature)) * np.arcsinh(np.sqrt(u_vect) * np.sinh(np.sqrt(-curvature)/2))
        return r    

    def _sample_uniformly(self, curvature, n_points):
        theta = 2 * np.pi * np.random.random_sample((n_points,))
        r = self._phi(curvature, np.random.random_sample((n_points,)))
        return np.stack((r,theta), axis = -1)
            
    def _geodesic_distance(self, curvature, x1 , x2):
        
        if curvature > 0:
            R = 1/np.sqrt(curvature)
            v1 = np.array([R * np.sin(x1[0]/R) * np.cos(x1[1]), 
                        R * np.sin(x1[0]/R) * np.sin(x1[1]),
                        R * np.cos(x1[0]/R)])
            
            v2 = np.array([R * np.sin(x2[0]/R) * np.cos(x2[1]), 
                        R * np.sin(x2[0]/R) * np.sin(x2[1]),
                        R * np.cos(x2[0]/R)])

            
            dist = R * np.arctan2(np.linalg.norm(np.cross(v1,v2)), (v1*v2).sum())
        
        elif curvature == 0:
            v1 = np.array([x1[0]*np.cos(x1[1]), x1[0]*np.sin(x1[1])])
            v2 = np.array([x2[0]*np.cos(x2[1]), x2[0]*np.sin(x2[1])])
            dist = np.linalg.norm( (v1 - v2) )  
        
        elif curvature < 0:
            R = 1/np.sqrt(-curvature)
            z = np.array([ np.tanh(x1[0]/(2 * R)) * np.cos(x1[1]),
                        np.tanh(x1[0]/(2 * R)) * np.sin(x1[1])])
            w = np.array([np.tanh(x2[0]/(2 * R)) * np.cos(x2[1]),
                        np.tanh(x2[0]/(2 * R)) * np.sin(x2[1])])
            temp = np.linalg.norm([(z*w).sum() - 1, np.linalg.det([z,w]) + 1])
            dist = 2 * R * np.arctanh(np.linalg.norm(z - w)/temp) 
            
        return dist

    def _compute_distance_matrix(self, curvature, n_points):
        metric = lambda x1, x2 : self._geodesic_distance(curvature, x1 , x2)
        samples = self._sample_uniformly(curvature, n_points)
        return pairwise_distances(samples, metric = metric)
    
    def _compute_diagrams(self):  
        """ This functions outputs the persistence diagrams in homological dimensions given by
        the homology dimensions, 
        for a list datasets obtained by uniformly sampling n_points for each elements in curvatures
        """
        
        distance_matrices = []
        VR = VietorisRipsPersistence(homology_dimensions=self._homology_dimensions,
                                     metric = 'precomputed',  n_jobs=self._n_jobs)
        # for curvature in self._curvatures:
        #     distance_matrix = self._compute_distance_matrix(curvature,
        #                                     self._num_points_per_sampling)
        #     distance_matrices.append(distance_matrix)
        
        def process(i):
            return self._compute_distance_matrix(self._curvatures[i],
                                            self._num_points_per_sampling)
        
        distance_matrices = Parallel(n_jobs=self._n_jobs)(delayed(process)(i)
                                                for i in range(len(self._curvatures)))
        
        self._diagrams = VR.fit_transform(distance_matrices)

    def _persistence_diagrams_to_one_hot(self, persistence_diagrams):
        """ Convert homology dimension to one-hot encoding

        Args:
            persistence_diagrams ([np.array]): persistence diagram with categorical
                homology dimension.

        Returns:
            [np.array]: persistent diagram with one-hot encoded homology dimension.
        """
        return np.concatenate(
            (
                self._diagrams[:, :, :2],  # point coordinates
                (np.eye(len(self._homology_dimensions))  # type: ignore
                [self._diagrams[:, :, -1].astype(np.int32)]),
            ),
            axis=-1)

    def get_diagrams(self):
        return self._persistence_diagrams_to_one_hot(self._diagrams)
    
    def get_curvatures(self):
        return self._curvatures
    
    def get_dataloader(self, **dataloaders_kwargs):
        DataLoader(TensorDataset(self._diagrams, self._curvatures),
                   **dataloaders_kwargs)