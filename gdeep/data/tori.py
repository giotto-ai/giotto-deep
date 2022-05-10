import math
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_blobs


class Rotation():
    """Class for rotations
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
        rotation_matrix = np.identity(3)
        rotation_matrix[self._axis_0, self._axis_0]\
            = math.cos(self._angle)
        rotation_matrix[self._axis_1, self._axis_1]\
            = math.cos(self._angle)
        rotation_matrix[self._axis_1, self._axis_0]\
            = math.sin(self._angle)
        rotation_matrix[self._axis_0, self._axis_1]\
            = -math.sin(self._angle)
        return rotation_matrix


class GenericDataset(Dataset):
    """This class is the base class for the tori-datasets

    Args:
        data (Tensor):
            tensor with first dimension
            the number of samples
        taregts (list):
            list of labels
        transform (Callable):
            act on the single images
        target_transform (Callable):
            act on the single label
    """

    def __init__(self, data, targets,
                 transform=None, target_transform=None):
        self.targets = targets
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = [image.to(torch.float), label.to(torch.long)]
        return sample


class CreateToriDataset:
    """This class is used to generate data loaders for the
    family of tori-datasets

    Args:
        name (string):
            name of the torus dataset to generate
    """
    def __init__(self, name):
        self.name = name

    def generate_dataset(self, **kwargs):
        """This method is the maion method of the class.
        It generates the ToriDatasset class depending on
        the self.name"""
        if self.name == "DoubleTori":
            tup = self._make_two_tori_dataset(**kwargs)
            return GenericDataset(*tup, target_transform=torch.tensor)
        elif self.name == "Blobs":
            tup = self._make_blobs(**kwargs)
            return GenericDataset(*tup, target_transform=torch.tensor)
        else:
            tup = self._make_entangled_tori_dataset(**self.kwargs)
            return GenericDataset(*tup, target_transform=torch.tensor)

    @staticmethod
    def _make_torus_point_cloud(label: int, n_points: int, noise: float,
                                rotation: Rotation, base_point: np.ndarray,
                                radius1: float = 1., radius2: float = 1.):
        """Generate point cloud of a single torus using
        2 radii for its definition

        Args:
            label (int):
                label of the data points
            n_points (int):
                number of sample points for each direction
            noise (float):
                noise
            rotation:
                Rotation
            base_point (np.array):
                center of the torus
            radius1 (float):
                radius of torus 1
            radius2 (float):
                radius of torus 2

        Returns:
            (np.array, np.array):
                data_points, labels
        """
        torus_point_clouds = np.asarray(
            [
                [
                    (radius1 + radius2 * np.cos(s)) * np.cos(t) +
                    noise * (np.random.rand(1)[0] - 0.5),
                    (radius1 + radius2 * np.cos(s)) * np.sin(t) +
                    noise * (np.random.rand(1)[0] - 0.5),
                    radius2 * np.sin(s) + noise *
                    (np.random.rand(1)[0] - 0.5),
                ]
                for t in range(n_points)
                for s in range(n_points)])

        torus_point_clouds = np.einsum("ij,kj->ki",
                                       rotation.rotation_matrix(),
                                       torus_point_clouds)

        torus_point_clouds += base_point

        # label tori with 2
        torus_labels = label * np.ones(n_points ** 2)

        return torus_point_clouds, torus_labels
    
    @staticmethod
    def _make_torus_point_cloud_fixed_radius(label: int, n_points: int,
                                noise: float,
                                rotation: Rotation,
                                base_point: np.ndarray,
                                radius: float = 1.):
        """Generate point cloud of a single torus

        Args:
            label (int):
                label of the data points
            n_points (int):
                number of sample points for each direction
            noise (float):
                noise
            rotation (Rotation):
                rotation class
            base_point (np.array):
                center of the torus
            radius (float):
                radius of the torus

        Returns:
            (np.array, np.array):
                data_points, labels
        """
        return CreateToriDataset.\
            _make_torus_point_cloud(label, n_points, noise,
                                        rotation, base_point, radius,
                                        radius)

    def _make_two_tori_dataset(self, entangled: bool = True,
                               n_pts: int = 10) -> tuple:
        """Generates pandas Dataframe of two tori in 3D. The labels correspond to
        the different Tori.

        Args:
            entangled (bool, optional):
                Either entangled or unentangled tori. Defaults to True.

        Returns:
            (tuple):
                the tuple for data and labels
        """
        torus_point_cloud = {}
        torus_labels = {}
        if entangled:
            torus_point_cloud[0], torus_labels[0] = \
                self._make_torus_point_cloud_fixed_radius(0, n_pts, 0.0,
                                             Rotation(1, 2,
                                                      math.pi/2),
                                             np.array([[0, 0, 0]]),
                                             radius=.3)
            torus_point_cloud[1], torus_labels[1] = \
                self._make_torus_point_cloud_fixed_radius(1, n_pts, 0.0,
                                             Rotation(1, 2, 0),
                                             np.array([[2, 0, 0]]),
                                             radius=.3)
        else:
            torus_point_cloud[0], torus_labels[0] = \
                self._make_torus_point_cloud_fixed_radius(0, n_pts, 0.0,
                                             Rotation(1, 2,
                                                      math.pi/2),
                                             np.array([[0, 0, 0]]),
                                             radius=.3)
            torus_point_cloud[1], torus_labels[1] = \
                self._make_torus_point_cloud_fixed_radius(1, n_pts, 0.0,
                                             Rotation(1, 2, 0),
                                             np.array([[6, 0, 0]]),
                                             radius=.3)

        tori_point_cloud = np.concatenate((torus_point_cloud[0],
                                           torus_point_cloud[1]),
                                          axis=0)
        tori_labels = np.concatenate((torus_labels[0],
                                      torus_labels[1]),
                                     axis=0)
        return torch.from_numpy(tori_point_cloud), tori_labels

    def _make_entangled_tori_dataset(self, m: int = 2,
                                     n_pts: int = 10) -> tuple:
        """Generates pandas Dataframe of m x m x m tori in 3D.
        The labels correspond to the different Tori.

        Args:
            m (int, defalut=2):
                Number of entangled tori per axis
            n_pts (int):
                Number of points per torus

        Returns:
            (tuple):
                the tuple for data and labels
        """
        data1, labels1 = self._make_torus_point_cloud(0, n_pts,
                                                       0.1, Rotation(0, 1, 0),
                                                       np.array([0, 0, 0]), 
                                                       2., .5)
        data2, labels2 = self._make_torus_point_cloud(1, n_pts, 0.1,
                                                       Rotation(1, 2, 180),
                                                       np.array([2, 0, 0]), 
                                                       2., .5)

        translations = [[i*10, j*10, k*10] for i in range(m) for j
                        in range(m) for k in range(m)]

        tori_entangled = np.append(data1, data2, axis=0)
        labels = np.append(labels1, labels2)
        lab = labels
        data = tori_entangled

        for t in translations:
            if t != [0, 0, 0]:
                data = np.append(data, np.add(tori_entangled, t), axis=0)
                lab = np.append(lab, labels)
        return torch.from_numpy(data), lab

    @staticmethod
    def _make_blobs(m: int = 3, n_pts: int = 200) -> tuple:
        """Generates blobs in 3D.
        The labels correspond to the different blob.

        Args:
            m (int, defalut=2):
                Number of entangled tori per axis
            n_pts (int):
                Number of points per torus

        Returns:
            (tuple):
                the tuple for data and labels
        """
        data, lab = make_blobs(n_samples=n_pts, centers=m,
                               n_features=3, random_state=42)
        return torch.from_numpy(data), lab
