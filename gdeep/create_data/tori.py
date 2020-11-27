import math
import numpy as np

class Rotation():
    """ Rotation in 3 dimensions
    axis_0 and axis_1 are the standard basis vectors that
    span the plane the rotation takes place and angle is
    the angle of the rotation.
    Args:
            axis_0 (int): first basis vector of plane
            axis_1 (int): second basis vector of plane
            angle (float): rotation arrow
    """
    def __init__(self, axis_0, axis_1, angle):
        self._axis_0 = axis_0
        self._axis_1 = axis_1
        self._angle = angle

    def return_axis(self, idx):
        return eval('self._axis_'+str(idx))
    
    def return_angle(self):
        return self._angle

    def rotation_matrix(self):
        """ Returns the rotation matrix corresponding
        to the rotation

        Returns:
            ndarray: ratation matrix of shape (3, 3)
        """
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
