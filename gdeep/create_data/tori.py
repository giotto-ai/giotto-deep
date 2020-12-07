import math
import numpy as np
import pandas as pd

class Rotation():
    def __init__(self, axis_0, axis_1, angle):
        self._axis_0 = axis_0
        self._axis_1 = axis_1
        self._angle = angle

    def return_axis(self, idx):
        return eval('self._axis_'+str(idx))
    
    def return_angle(self):
        return self._angle

    def rotation_matrix(self):
        rotation_matrix = np.identity(3)
        rotation_matrix[self._axis_0,self._axis_0]\
            = math.cos(self._angle)
        rotation_matrix[self._axis_1,self._axis_1]\
            = math.cos(self._angle)
        rotation_matrix[self._axis_1,self._axis_0]\
            = math.sin(self._angle)
        rotation_matrix[self._axis_0,self._axis_1]\
            = -math.sin(self._angle)
        return rotation_matrix


def make_torus_point_cloud(label: int, n_points: int, noise: float,\
    rotation: Rotation, base_point: np.array, radius: float=1.):
    """Generate point cloud of a torus

    Args:
        label (int): label of the data points
        n_points (int): number of sample points for each direction
        noise (float): noise
        rotation: Rotation
        base_point (np.array): center of the torus
        radius: float

    Returns:
        (np.array, np.array): data_points, labels
    """
    torus_point_clouds = np.asarray(
            [
                [
                    (2 + radius*np.cos(s)) * np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                    (2 + radius*np.cos(s)) * np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                    radius*np.sin(s) + noise * (np.random.rand(1)[0] - 0.5),
                ]
                for t in range(n_points)
                for s in range(n_points)
            ]
        )
    
    torus_point_clouds = np.einsum("ij,kj->ki",  rotation.rotation_matrix(), torus_point_clouds)

    torus_point_clouds += base_point

    # label tori with 2
    torus_labels = label * np.ones(n_points**2)

    return torus_point_clouds, torus_labels


def make_torus_dataset(entangled: bool=True)->pd.core.frame.DataFrame:
    """Generates pandas Dataframe of two tori in 3D. The labels correspond to
    the different Tori.

    Args:
        entangled (bool, optional): Either entangled or unentangled tori. Defaults to True.

    Returns:
        pd.core.frame.DataFrame: pandas dataframe with columns `x1, x2, x3, label`.
    """
    torus_point_cloud = {}
    torus_labels = {}
    if entangled:
        torus_point_cloud[0], torus_labels[0] = make_torus_point_cloud(0, 50, 0.0,\
            Rotation(1,2,math.pi/2), np.array([[0,0,0]]), radius=.3)
        torus_point_cloud[1], torus_labels[1]  = make_torus_point_cloud(1, 50, 0.0,\
            Rotation(1,2,0), np.array([[2,0,0]]), radius=.3)
    else:
        torus_point_cloud[0], torus_labels[0] = make_torus_point_cloud(0, 50, 0.0,\
            Rotation(1,2,math.pi/2), np.array([[0,0,0]]), radius=.3)
        torus_point_cloud[1], torus_labels[1]  = make_torus_point_cloud(1, 50, 0.0,\
            Rotation(1,2,0), np.array([[6,0,0]]), radius=.3)

    tori_point_cloud = np.concatenate((torus_point_cloud[0],\
                                torus_point_cloud[1]), axis=0)
    tori_labels = np.concatenate((torus_labels[0],\
                                torus_labels[1]), axis=0)
    #print(f'tori_point_cloud: {tori_point_cloud.shape}, tori_labels: {tori_labels.shape}')
    return pd.DataFrame(np.concatenate(
        (tori_point_cloud, tori_labels.reshape((-1,1))),
         axis=-1),
         columns = ["x1", "x2", "x3", "label"])