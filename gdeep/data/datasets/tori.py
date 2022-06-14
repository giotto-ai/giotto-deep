import math
from typing import Tuple, Union, List

import numpy as np
from sklearn.datasets import make_blobs
import torch
from torch.utils.data import Dataset


Tensor = torch.Tensor
Array = np.ndarray


class Rotation():
    """Class for rotations
    """
    def __init__(self, axis_0: int,
                 axis_1: int,
                 angle: float) -> None:
        self._axis_0 = axis_0
        self._axis_1 = axis_1
        self._angle = angle

    def return_axis(self, idx: int):
        return eval('self._axis_'+str(idx))

    def return_angle(self) -> float:
        return self._angle

    def rotation_matrix(self) -> Array:
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


class ToriDataset(Dataset[Tuple[Tensor, Tensor]]):
    """This class is used to generate data loaders for the
    family of tori-datasets

    Args:
        name:
            name of the torus dataset to generate
    """
    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        if self.name == "DoubleTori":
            self.out_tuple = self._make_two_tori_dataset(**kwargs)
        elif self.name == "Blobs":
            self.out_tuple = self._make_blobs(**kwargs)
        else:
            self.out_tuple = self._make_entangled_tori_dataset(**kwargs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """This method is the main method of the class.
        It extracts the i-the element
        """

        return self.out_tuple[0][idx], torch.tensor(self.out_tuple[1][idx]).to(torch.long)

    def __len__(self) -> int:
        return len(self.out_tuple[0])

    @staticmethod
    def _make_torus_point_cloud(label: int, 
                                n_points: int,
                                noise: float,
                                rotation: Rotation,
                                base_point: Array,
                                radius: float=1.) -> Tuple[Array, Array]:
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
        torus_point_clouds = \
            np.asarray(
                [
                    [
                     (2 + radius*np.cos(s)) * np.cos(t) + noise *
                     (np.random.rand(1)[0] - 0.5),
                     (2 + radius*np.cos(s)) * np.sin(t) + noise *
                     (np.random.rand(1)[0] - 0.5),
                     radius*np.sin(s) + noise *
                     (np.random.rand(1)[0] - 0.5),
                    ]
                    for t in range(n_points)
                    for s in range(n_points)])
        
        torus_point_clouds = np.einsum("ij,kj->ki",  rotation.rotation_matrix(), torus_point_clouds)

        torus_point_clouds += base_point

        # label tori with 2
        torus_labels = label * np.ones(n_points**2)

        return torus_point_clouds, torus_labels

    @staticmethod
    def _make_torus_point_cloud2(label: int, n_points: int, noise: float,
                                rotation: Rotation, base_point: Union[Array, List],
                                radius1: float = 1., radius2: float = 1.) -> Tuple[Array, Array]:
        """Generate point cloud of a single torus using
        2 radii for its definition

        Args:
            label :
                label of the data points
            n_points :
                number of sample points for each direction
            noise :
                noise
            rotation:
                Rotation
            base_point :
                center of the torus
            radius1 :
                radius of torus 1
            radius2 :
                radius of torus 2

        Returns:
            (np.ndarray, np.ndarray):
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

    def _make_two_tori_dataset(self, entangled: bool = True,
                               n_pts: int = 10) -> Tuple[Tensor, Array]:
        """Generates pandas Dataframe of two tori in 3D. The labels correspond to
        the different Tori.

        Args:
            entangled :
                Either entangled or unentangled tori. Defaults to True.

        Returns:
            (tuple):
                the tuple for data and labels
        """
        torus_point_cloud = {}
        torus_labels = {}
        if entangled:
            torus_point_cloud[0], torus_labels[0] = \
                self._make_torus_point_cloud(0, n_pts, 0.0,
                                             Rotation(1, 2,
                                                      math.pi/2),
                                             np.array([[0, 0, 0]]),
                                             radius=.3)
            torus_point_cloud[1], torus_labels[1] = \
                self._make_torus_point_cloud(1, n_pts, 0.0,
                                             Rotation(1, 2, 0),
                                             np.array([[2, 0, 0]]),
                                             radius=.3)
        else:
            torus_point_cloud[0], torus_labels[0] = \
                self._make_torus_point_cloud(0, n_pts, 0.0,
                                             Rotation(1, 2,
                                                      math.pi/2),
                                             np.array([[0, 0, 0]]),
                                             radius=.3)
            torus_point_cloud[1], torus_labels[1] = \
                self._make_torus_point_cloud(1, n_pts, 0.0,
                                             Rotation(1, 2, 0),
                                             np.array([[6, 0, 0]]),
                                             radius=.3)

        tori_point_cloud = np.concatenate((torus_point_cloud[0],
                                           torus_point_cloud[1]),
                                          axis=0)
        tori_labels = np.concatenate((torus_labels[0],
                                      torus_labels[1]),
                                     axis=0)
        return torch.from_numpy(tori_point_cloud).to(torch.float32), tori_labels

    def _make_entangled_tori_dataset(self, m: int = 2,
                                     n_pts: int = 10) -> Tuple[Tensor, Array]:
        """Generates pandas Dataframe of m x m x m tori in 3D.
        The labels correspond to the different Tori.

        Args:
            m :
                Number of entangled tori per axis
            n_pts :
                Number of points per torus

        Returns:
            (tuple):
                the tuple for data and labels
        """
        data1, labels1 = self._make_torus_point_cloud2(0, n_pts,
                                                       0.1, Rotation(0, 1, 0),
                                                       [0, 0, 0], 2., .5)
        data2, labels2 = self._make_torus_point_cloud2(1, n_pts, 0.1,
                                                       Rotation(1, 2, 180),
                                                       [2, 0, 0], 2., .5)

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
        return torch.from_numpy(data).to(torch.float32), lab

    @staticmethod
    def _make_blobs(m: int = 3, n_pts: int = 200) -> Tuple[Tensor, Array]:
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
        return torch.from_numpy(data).to(torch.float32), lab
