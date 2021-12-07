import numpy as np
from sklearn.metrics import pairwise_distances
from gtda.homology import VietorisRipsPersistence
from torch.utils.data import DataLoader, TensorDataset  # type: ignore


#from numba import cuda
import numba as nb
from math import tanh, cos, sin, sqrt, atanh, atan2

USE_64 = True

if USE_64:
    bits = 64
    np_type = np.float64
else:
    bits = 32
    np_type = np.float32
    

# @cuda.jit("void(float{}[:, :], float{}[:, :], float{})".format(bits, bits, bits))
# def distance_matrix(mat, out, curvature):
#     m = mat.shape[0]
#     i, j = cuda.grid(2)
#     if i < m and j < m:
#         if curvature > 0:
#             R = 1.0/sqrt(curvature)
#             z0 = R * sin(mat[i, 0] / R) * cos(mat[i, 1])
#             z1 = R * sin(mat[i, 0] / R) * sin(mat[i, 1])
#             z2 = R * cos(mat[i, 0] / R)
            
#             w0 = R * sin(mat[j, 0] / R) * cos(mat[j, 1])
#             w1 = R * sin(mat[j, 0] / R) * sin(mat[j, 1])
#             w2 = R * cos(mat[j, 0] / R)
            
#             cross0 = z1 * w2 - z2 * w1
#             cross1 = z2 * w0 - z0 * w2
#             cross2 = z0 * w1 - z1 * w0
            
#             out[i, j] = R * atan2(sqrt(cross0 * cross0 + cross1 * cross1 + cross2 * cross2),
#                                   z0 * w0 + z1 * w1 + z2 * w2)
        
#         if curvature < 0:
#             R = 1.0/sqrt(-curvature)
#             z0 = tanh(mat[i, 0]/(2.0 * R)) * cos(mat[i, 1])
#             z1 = tanh(mat[i, 0]/(2.0 * R)) * sin(mat[i, 1])
#             w0 = tanh(mat[j, 0]/(2.0 * R)) * cos(mat[j, 1])
#             w1 = tanh(mat[j, 0]/(2.0 * R)) * sin(mat[j, 1])
            
#             temp0 = z0 * w0 + z1 * w1 - 1.0
#             temp1 = z0 * w1 - z1 * w0 + 1.0
#             temp = sqrt(temp0 * temp0 + temp1 * temp1)
#             x = sqrt((z0 - w0) * (z0 - w0) + (z1 - w1) * (z1 - w1))/temp
#             out[i, j] = 2.0 * R * atanh(x)
            
#         if curvature == 0.0:  # it does not make sense to compare floats
#             z0 = mat[i, 0] * cos(mat[i, 1])
#             z1 = mat[i, 0] * sin(mat[i, 1])
            
#             w0 = mat[j, 0] * cos(mat[j, 1])
#             w1 = mat[j, 0] * sin(mat[j, 1])
            
#             out[i, j] = sqrt((z0 - w0) * (z0 - w0) + (z1 - w1) * (z1 - w1))

# def gpu_dist_matrix(mat, curvature):
#     rows = mat.shape[0]

#     block_dim = (16, 16)
#     grid_dim = (int(rows/block_dim[0] + 1), int(rows/block_dim[1] + 1))

#     stream = cuda.stream()
#     mat2 = cuda.to_device(np.asarray(mat, dtype=np_type), stream=stream)
#     out2 = cuda.device_array((rows, rows))
#     distance_matrix[grid_dim, block_dim](mat2, out2, curvature)
#     out = out2.copy_to_host(stream=stream)

#     return out

@nb.njit
def cpu_dist_matrix(mat, curvature):
    m = mat.shape[0]

    out = np.empty((m, m), dtype=np_type)  # corrected dtype


    for i in range(m):
        out[i, i] = 0.0
        for j in range(i+1, m):
            if curvature > 0:
                R = 1.0/sqrt(curvature)
                z0 = R * sin(mat[i, 0] / R) * cos(mat[i, 1])
                z1 = R * sin(mat[i, 0] / R) * sin(mat[i, 1])
                z2 = R * cos(mat[i, 0] / R)
                
                w0 = R * sin(mat[j, 0] / R) * cos(mat[j, 1])
                w1 = R * sin(mat[j, 0] / R) * sin(mat[j, 1])
                w2 = R * cos(mat[j, 0] / R)
                
                cross0 = z1 * w2 - z2 * w1
                cross1 = z2 * w0 - z0 * w2
                cross2 = z0 * w1 - z1 * w0
                
                out[i, j] = R * atan2(sqrt(cross0 * cross0 + cross1 * cross1 + cross2 * cross2),
                                    z0 * w0 + z1 * w1 + z2 * w2)
            
            if curvature < 0:
                R = 1.0/sqrt(-curvature)
                z0 = tanh(mat[i, 0]/(2.0 * R)) * cos(mat[i, 1])
                z1 = tanh(mat[i, 0]/(2.0 * R)) * sin(mat[i, 1])
                w0 = tanh(mat[j, 0]/(2.0 * R)) * cos(mat[j, 1])
                w1 = tanh(mat[j, 0]/(2.0 * R)) * sin(mat[j, 1])
                
                temp0 = z0 * w0 + z1 * w1 - 1.0
                temp1 = z0 * w1 - z1 * w0 + 1.0
                temp = sqrt(temp0 * temp0 + temp1 * temp1)
                x = sqrt((z0 - w0) * (z0 - w0) + (z1 - w1) * (z1 - w1))/temp
                out[i, j] = 2.0 * R * atanh(x)
                
            if curvature == 0.0:  # it does not make sense to compare floats
                z0 = mat[i, 0] * cos(mat[i, 1])
                z1 = mat[i, 0] * sin(mat[i, 1])
                
                w0 = mat[j, 0] * cos(mat[j, 1])
                w1 = mat[j, 0] * sin(mat[j, 1])
                
                out[i, j] = sqrt((z0 - w0) * (z0 - w0) + (z1 - w1) * (z1 - w1))
            out[j, i] = out[i, j]

    return out



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
            
    # def _geodesic_distance(self, curvature, x1 , x2):
        
    #     if curvature > 0:
    #         R = 1/np.sqrt(curvature)
    #         v1 = np.array([R * np.sin(x1[0]/R) * np.cos(x1[1]), 
    #                     R * np.sin(x1[0]/R) * np.sin(x1[1]),
    #                     R * np.cos(x1[0]/R)])
            
    #         v2 = np.array([R * np.sin(x2[0]/R) * np.cos(x2[1]), 
    #                     R * np.sin(x2[0]/R) * np.sin(x2[1]),
    #                     R * np.cos(x2[0]/R)])

            
    #         dist = R * np.arctan2(np.linalg.norm(np.cross(v1,v2)), (v1*v2).sum())
        
    #     elif curvature == 0:
    #         v1 = np.array([x1[0]*np.cos(x1[1]), x1[0]*np.sin(x1[1])])
    #         v2 = np.array([x2[0]*np.cos(x2[1]), x2[0]*np.sin(x2[1])])
    #         dist = np.linalg.norm( (v1 - v2) )  
        
    #     elif curvature < 0:
    #         R = 1/np.sqrt(-curvature)
    #         z = np.array([ np.tanh(x1[0]/(2 * R)) * np.cos(x1[1]),
    #                     np.tanh(x1[0]/(2 * R)) * np.sin(x1[1])])
    #         w = np.array([np.tanh(x2[0]/(2 * R)) * np.cos(x2[1]),
    #                     np.tanh(x2[0]/(2 * R)) * np.sin(x2[1])])
    #         temp = np.linalg.norm([(z*w).sum() - 1, np.linalg.det([z,w]) + 1])
    #         dist = 2 * R * np.arctanh(np.linalg.norm(z - w)/temp) 
            
    #     return dist

    # def _compute_distance_matrix(self, curvature, n_points):
    #     metric = lambda x1, x2 : self._geodesic_distance(curvature, x1 , x2)
    #     samples = self._sample_uniformly(curvature, n_points)
    #     return pairwise_distances(samples, metric = metric)
    
    def _compute_distance_matrix(self, curvature, n_points):
        samples = self._sample_uniformly(curvature, n_points)
        return cpu_dist_matrix(samples, curvature)
    
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
        
        distance_matrices = [process(i) for i in range(len(self._curvatures))]
        
        self._diagrams = VR.fit_transform(distance_matrices)

    def _persistence_diagrams_to_one_hot(self):
        """ Convert homology dimension to one-hot encoding
        """
        return np.concatenate(
            (
                self._diagrams[:, :, :2],  # point coordinates
                (np.eye(len(self._homology_dimensions))  # type: ignore
                [self._diagrams[:, :, -1].astype(np.int32)]),
            ),
            axis=-1)

    def get_diagrams(self):
        if len(self._homology_dimensions) > 1:
            return self._persistence_diagrams_to_one_hot()
        else:
            return self._diagrams
    
    def get_curvatures(self):
        return self._curvatures
    
    def get_dataloader(self, **dataloaders_kwargs):
        DataLoader(TensorDataset(self._diagrams, self._curvatures),
                   **dataloaders_kwargs)