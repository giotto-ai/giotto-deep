from typing import List
import numpy as np  # type: ignore
import torch
import multiprocessing
import torch
from einops import rearrange  # type: ignore
from gtda.homology import WeakAlphaPersistence  # type: ignore

def generate_orbit_parallel(
    num_classes,
    num_orbits,
    num_pts_per_orbit: int = 100,
    parameters: List[float] = [1.0],
    ) -> np.ndarray:
    """Generate sequence of points of a dynamical system
    in a parallel manner.

    Args:
        num_classes (int): number of classes of dynamical systems.
        num_orbits (int): number of orbits of dynamical system per class.
        num_pts_per_orbit (int, optional): Number of points to generate.
            Defaults to 100.
        parameter (List[float], optional): List of parameters of the dynamical
            system.
            Defaults to [1.0].

    Returns:
        np.ndarray: Array of sampled points of the dynamical system.
    """
    try:
        for parameter in parameters:
            assert parameter > 0
    except AssertionError:
        print('Parameter must be greater than 0')
    try:
        assert num_pts_per_orbit > 0
    except AssertionError:
        print('num_pts_per_orbit must be greater than 0')

    x = np.zeros((
                    num_classes,
                    num_orbits,
                    num_pts_per_orbit,
                    2
                ))

    # generate dataset
    for cidx, p in enumerate(parameters):  # type: ignore
        x[cidx, :, 0, :] = np.random.rand(num_orbits, 2)  # type: ignore

        for i in range(1, num_pts_per_orbit):  # type: ignore
            x_cur = x[cidx, :, i - 1, 0]
            y_cur = x[cidx, :, i - 1, 1]

            x[cidx, :, i, 0] = (x_cur + p * y_cur * (1. - y_cur)) % 1
            x_next = x[cidx, :, i, 0]
            x[cidx, :, i, 1] = (y_cur + p * x_next * (1. - x_next)) % 1

    assert(not np.allclose(x[0, 0], x[0, 1]))
    return x

def create_pd_orbits(
    orbits,
    num_classes,
    homology_dimensions = (0, 1),
    n_jobs = multiprocessing.cpu_count(),
    ) -> torch.Tensor:
    """ Computes the weak alpha persistence of the orbit data clouds.

    Args:
        orbits (np.array): Orbits of shape [n_points, 2]
        homology_dimensions (tuple, optional): Dimensions to compute the
            persistence diagrams.
            Defaults to (0, 1).
        n_jobs (int, optional): Number of cpus to use for parallel computation.
            Defaults to multiprocessing.cpu_count().

    Returns:
        np.array: Array of persistence diagrams of shape
            [num_classes, num_orbits, num_persistence_points, 3].
            In the last dimension the first two values are the coordinates of
            the points in the persistence diagrams and the third is the
            homology dimension.
    """
    # compute weak alpha persistence
    wap = WeakAlphaPersistence(
                        homology_dimensions=homology_dimensions,
                        n_jobs=n_jobs
                        )
    # c: class, o: orbit, p: point, d: dimension
    orbits_stack = rearrange(orbits, 'c o p d -> (c o) p d')  # stack classes
    diagrams = wap.fit_transform(orbits_stack)
    # shape: (num_classes * n_samples, n_features, 3)

    # combine class and orbit dimensions
    diagrams = rearrange(
                            diagrams,
                            '(c o) p d -> c o p d',
                            c=num_classes  # type: ignore
                        )
    return diagrams

def convert_pd_orbits_to_tensor(
    diagrams: np.ndarray,
    num_classes: int,
    num_orbits: int,
    num_homology_dimensions: int = 2,
    ):
    """[summary]

    Args:
        diagrams (np.ndarray): [description]
        num_classes (int): [description]
        num_orbits (int): [description]
        num_homology_dimensions (int, optional): [description]. Defaults to 2.

    Returns:
        torch.tensor: tuple of
            1.
            2. label tensor.
    """
    # c: class, o: orbit, p: point in persistence diagram,
    # d: coordinates + homology dimension
    x = rearrange(
                    diagrams,
                    'c o p d -> (c o) p d',
                    c=num_classes  # type: ignore
                )
    # convert homology dimension to one-hot encoding
    x = np.concatenate(
        (
            x[:, :, :2],
            (np.eye(num_homology_dimensions)  # type: ignore
             [x[:, :, -1].astype(np.int32)]),
        ),
        axis=-1)
    # convert from [orbit, sequence_length, feature] to
    # [orbit, feature, sequence_length] to fit to the
    # input_shape of `SmallSetTransformer`
    # x = rearrange(x, 'o s f -> o f s')

    # generate labels
    y_list = []
    for i in range(num_classes):  # type: ignore
        y_list += [i] * num_orbits  # type: ignore

    y = np.array(y_list)

    # load dataset to PyTorch dataloader

    x_tensor = torch.Tensor(x)
    y_tensor = torch.Tensor(y).long()

    return x_tensor, y_tensor