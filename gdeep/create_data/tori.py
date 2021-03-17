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
    rotation: Rotation, base_point: np.array, radius: float = 1.):
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


<<<<<<< HEAD
def sample_torus_uniformely(n_samples: int = 100,
                            r: float = .3, R: float = 1.):
    """ Sample uniformly points on embedded torus where a
    distance R from the centre of the tube to the centre
    of the torus and a distance r from the centre of the
    tube to the surface of the tube with r≤R.
    cf. https://math.stackexchange.com/questions/2017079
    /uniform-random-points-on-a-torus

    Args:
        float (r, optional): [description]. Defaults to 1..
        float (R, optional): [description]. Defaults to .5.

    Returns:
        np.array: uniformely sampled points on torus of shape (n_samples, 3).
    """
    assert r <= R

    n_points = 0
    sample_points = np.zeros((n_samples, 3))

    while(n_points < n_samples):
        u, v, w = tuple(np.random.uniform(0., 1., 3))
        theta, psi = 2.*np.pi*u, 2.*np.pi*v
        if w <= (R + r * np.cos(theta))/(R + r):
            sample_points[n_points, :] = np.array([
                (R + r * np.cos(theta)) * np.cos(psi),
                (R + r * np.cos(theta)) * np.sin(psi),
                r * np.sin(theta)
            ])
            n_points += 1

    return sample_points

def sample_klein_bottle_uniformely(n_samples: int = 100,
                                   r: float = 0.75):
    """ Sample points on a klein bottle embedded in R^4
    uniformely with respect to the volumn of the restriced
    euclidian metric.
    https://corybrunson.github.io/2019/02/01/sampling/

    Args:
        n_samples (int, optional): number of sample points.
            Defaults to 100.
        r (float, optional): [description]. Defaults to 0.75.
    """
    assert r >= 0 and r <= 1

    def jacobian_klein(r):
        return lambda theta: r * np.sqrt((1 + r * np.cos(theta)) ^ 2 +
                                         (.5 * r * np.sin(theta)) ^ 2)

    jacobian_klein_vectorized = np.vectorize(jacobian_klein)
    
    def sample_klein_theta(n_samples, r):
        x = []
        while (len(x) < n_samples):
            theta = np.random.uniform(0, 2 * np.pi, n_samples)
            jacobian = jacobian_klein(r)
            jacobian_theta <- sapply(theta, jacobian)
            eta = np.random.uniform(n, 0, jacobian(0))
            x <- c(x, theta[jacobian_theta > eta])
        }
        return x
    d= None
=======
def make_torus_point_cloud2(label: int, n_points: int, noise: float, \
                            rotation: Rotation, base_point: np.array, radius1: float = 1., radius2: float = 1.):
    """Generate point cloud of a torus using 2 radii for its definition

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
                (radius1 + radius2 * np.cos(s)) * np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                (radius1 + radius2 * np.cos(s)) * np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                radius2 * np.sin(s) + noise * (np.random.rand(1)[0] - 0.5),
            ]
            for t in range(n_points)
            for s in range(n_points)
        ]
    )

    torus_point_clouds = np.einsum("ij,kj->ki", rotation.rotation_matrix(), torus_point_clouds)

    torus_point_clouds += base_point

    # label tori with 2
    torus_labels = label * np.ones(n_points ** 2)

    return torus_point_clouds, torus_labels
>>>>>>> master


def make_torus_dataset(entangled: bool = True) -> pd.core.frame.DataFrame:
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