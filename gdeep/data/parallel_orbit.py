from typing import List, Tuple
import numpy as np  # type: ignore
import torch
import multiprocessing
import torch
from einops import rearrange  # type: ignore
from gtda.homology import WeakAlphaPersistence  # type: ignore


class orbits_generator(object):
    """Generate

    Args:
        parameters ([type]): [description]
        num_classes ([type]): [description]
        num_orbits_per_class ([type]): number of orbits per class
        num_pts_per_orbit ([type]): [description]
        homology_dimensions ([type]): [description]
        validation_percentage (float, optional): [description]. Defaults to 0.
        test_percentage (float, optional): [description]. Defaults to 0.
        dynamical_system (str, optional): either use persistence paths
            convention ´pp_convention´ or the classical convention
            ´classical_convention´. Defaults to '´classical_convention´'.
    """
    def __init__(self,
             parameters: Tuple[float] = (2.5, 3.5, 4.0, 4.1, 4.3),
             num_classes: int = 5,
             num_orbits_per_class: int = 1_000,
             num_pts_per_orbit: int = 1_000,
             homology_dimensions = (0, 1),
             validation_percentage: float=0,
             test_percentage: float=0,
             dynamical_system: str='classical_convention',
             n_jobs: int=1,
             ) -> None:

        self.parameters = parameters
        self.num_classes = num_classes
        self.num_orbits_per_class = num_orbits_per_class
        self.num_pts_per_orbit = num_pts_per_orbit
        self.homology_dimension = homology_dimensions
        self.validation_percentage = validation_percentage
        self.test_percentage = test_percentage
        self.num_homology_dimensions = len(self.homology_dimensions)
        
        assert dynamical_system in ('classical_convention', 'pp_convention'),\
            f'{dynamical_system} is not supported'
        self.dynamical_system = dynamical_system
        
        # Initalize orbits array and persistence diagrams array with None.
        self.orbits = None
        self.labels = None
        self.persistence_diagrams = None
            
    def _generate_orbits(self) -> None:
        if self.orbits != None:
            return
        else:
            x = np.zeros((
                self.num_classes,  # type: ignore
                self.num_orbits,
                self.num_pts_per_orbit,
                2
            ))

            y = np.array([self.num_orbits * [c] for c in range(self.num_classes)])
            
            self.labels = y.reshape(-1)

            # generate dataset
            for cidx, p in enumerate(self.parameters):  # type: ignore
                x[cidx, :, 0, :] = np.random.rand(self.num_orbits, 2)  # type: ignore

                for i in range(1, self.num_pts_per_orbit):  # type: ignore
                    x_cur = x[cidx, :, i - 1, 0]
                    y_cur = x[cidx, :, i - 1, 1]

                    if self.dynamical_system == 'pp_convention':
                        x[cidx, :, i, 0] = (x_cur + p * y_cur * (1. - y_cur)) % 1
                        x[cidx, :, i, 1] = (y_cur + p * x_cur * (1. - x_cur)) % 1
                    else:
                        x[cidx, :, i, 0] = (x_cur + p * y_cur * (1. - y_cur)) % 1
                        x_next = x[cidx, :, i, 0]
                        x[cidx, :, i, 1] = (y_cur + p * x_next * (1. - x_next)) % 1
                        
            self.orbits = x.reshape((-1, self.num_pts_per_orbit, 2))
    
    def _compute_persistence_diagrams(self) -> None:
        """ Computes the weak alpha persistence of the orbit data clouds.

            The result is stored in member variable ´persistence_diagrams´.
            It has the shape
            [num_classes, num_orbits, num_persistence_points, 3].
            In the last dimension the first two values are the coordinates of
            the points in the persistence diagrams and the third is the
            homology dimension.
        """
        if self.orbit == None:
            self._generate_orbits()
        
        # compute weak alpha persistence
        wap = WeakAlphaPersistence(
                            homology_dimensions=self.homology_dimensions,
                            n_jobs=self.n_jobs
                            )
        # c: class, o: orbit, p: point, d: dimension
        orbits_stack = rearrange(self.orbits, 'c o p d -> (c o) p d')  # stack classes
        diagrams = wap.fit_transform(orbits_stack)
        # shape: (num_classes * n_samples, n_features, 3)

        # combine class and orbit dimensions
        self.persistence_diagrams = rearrange(
                                diagrams,
                                '(c o) p d -> c o p d',
                                c=num_classes  # type: ignore
                            )
    
    def get_point_cloud(self) -> np.ndarray:
        if self.orbits == None:
            self._generate_orbits()
        return self.orbits
    
    def get_persistence_diagrams(self) -> np.ndarray:
        if self.persistence_diagrams is None:
            _compute_persistence_diagrams()
            
        return self.persistence_diagrams
    
    def _split_data(self, validation_size, test_size) -> None:
        X_train, X_test, y_train, y_test 
            = train_test_split(, self.labels, test_size=test_size, random_state=1)

        X_train, X_val, y_train, y_val 
            = train_test_split(X_train, y_train, test_size=, random_state=1)
    
    def get_point_cloud_dataset(self):
        pass


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