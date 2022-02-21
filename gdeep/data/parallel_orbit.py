from typing import List, Sequence, Tuple, Union
import numpy as np  # type: ignore
import multiprocessing
import torch  # type: ignore
from torch.utils.data import DataLoader, TensorDataset  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from einops import rearrange  # type: ignore
from gtda.homology import WeakAlphaPersistence  # type: ignore
from sympy import N, S

class DataLoaderKwargs(object):
    """Object to store keyword arguments for train, val, and test dataloaders
    """
    def __init__(self, *, train_kwargs, val_kwargs, test_kwargs):
        self.train_kwargs = train_kwargs
        self.val_kwargs = val_kwargs
        self.test_kwargs = test_kwargs
        
    def get_train_args(self):
        return self.train_kwargs
    
    def get_val_args(self):
        return self.val_kwargs
    
    def get_test_args(self):
        return self.test_kwargs
class OrbitsGenerator(object):
    """Generate Orbit dataset consistent of orbits defined by the dynamical system
    x[n+1] = x[n] + r * y[n] * (1  - y[n]) % 1
    y[n+1] = y[n] + r * x[n+1] * (1 - x[n+1]) % 1
    Note that there is an x[n+1] value in the second dimension.
    The parameter r is an hyperparameter and the classification task is to predict
    it given the orbit.
    By default r is chosen from (2.5, 3.5, 4.0, 4.1, 4.3).
    Args:
        parameters (Tuple[float]): 
            Hyperparameter of the dynamical systems.
        num_orbits_per_class (int): 
            number of orbits per class.
        num_pts_per_orbit (int): 
            number of points per orbit.
        homology_dimensions (Sequence[int]): 
            homology dimension of the persistence
            diagrams.
        validation_percentage (float, optional): 
            Percentage of the validation dataset.
            Defaults to 0.0.
        test_percentage (float, optional): 
            Percentage of the test dataset. Defaults to 0.0.
        dynamical_system (str, optional): 
            either use persistence paths
            convention ´pp_convention´ or the classical convention
            ´classical_convention´. Defaults to '´classical_convention´'.
        n_jobs (int, optional): 
            number of cpus to run the computation on. Defaults to 1.
    """
    def __init__(self,
                 parameters: Sequence[float] = (2.5, 3.5, 4.0, 4.1, 4.3),
                 num_orbits_per_class: int = 1_000,
                 num_pts_per_orbit: int = 1_000,
                 homology_dimensions: Sequence[int] = (0, 1),
                 validation_percentage: float = 0.0,
                 test_percentage: float = 0.0,
                 dynamical_system: str = 'classical_convention',
                 n_jobs: int = 1,
                 dtype: str = 'float32',
                 arbitrary_precision=False,
                 ) -> None:

        # Initialize member variables.
        self._parameters = parameters
        self._num_classes = len(self._parameters)
        self._num_orbits_per_class = num_orbits_per_class
        self._num_pts_per_orbit = num_pts_per_orbit
        self._homology_dimensions = homology_dimensions
        self._num_homology_dimensions = len(self._homology_dimensions)
        self._n_jobs = n_jobs
        # Assert that validation and testing parameters are valid
        assert (test_percentage >= 0.0 and validation_percentage >= 0.0
                and test_percentage + validation_percentage < 1.0)
        self._validation_percentage = validation_percentage
        self._test_percentage = test_percentage
        
        # Assert that convention for the dynamical system is valid
        assert dynamical_system in ('classical_convention', 'pp_convention'),\
            f'{dynamical_system} is not supported'
        self.dynamical_system = dynamical_system
        
        # Initialize orbits array and persistence diagrams array with None.
        self._orbits = None
        self._labels = None
        self._persistence_diagrams = None
        
        # Initialize the train, val, and test indices with None.
        self._train_idcs = None
        self._val_idcs = None
        self._test_idcs = None

        assert dtype in ('float32', 'float64', 'float128'), f"Type {dtype} is not supported."
        self._dtype = dtype

        self.arbitrary_precision = arbitrary_precision

    def orbits_from_array(self, orbits):
        assert (orbits.shape[0] == self._num_orbits_per_class * self._num_classes and
                orbits.shape[1] == self._num_pts_per_orbit and
                orbits.shape[2] == 2), "Array does not have the right shape."

        self._orbits = orbits

        y = np.array([self._num_orbits_per_class * [c]
                      for c in range(self._num_classes)])

        self._labels = y.reshape(-1)

    def _generate_orbits(self) -> None:
        """Generate the orbits for the dynamical system.
        """
        
        # If orbits are already computed do nothing
        if self._orbits is not None:
            return
        else:
            # Initialize the orbits array with zeros.
            x = np.zeros((
                self._num_classes,  # type: ignore
                self._num_orbits_per_class,
                self._num_pts_per_orbit,
                2
            ))
            # Initialize the labels array with the hyperparameter indices.
            y = np.array([self._num_orbits_per_class * [c]
                          for c in range(self._num_classes)])
            
            self._labels = y.reshape(-1).astype('int64')  # type: ignore
            # generate dataset
            for class_idx, p in enumerate(self._parameters):  # type: ignore
                x[class_idx, :, 0, :] = np.random.rand(self._num_orbits_per_class, 2)  # type: ignore

                if self.arbitrary_precision:
                    assert self.dynamical_system == 'classical_convention', "Only classical_convention implemented yet"
                    for orbit in range(self._num_orbits_per_class):
                        print(orbit)
                        x[class_idx, orbit, :, :] = self._orbit_high_precision(
                                                        x_init=x[class_idx, orbit, 0, :],
                                                        rho=p,
                                                        num_points=self._num_pts_per_orbit,
                                                        precision=600)

                else:

                    for i in range(1, self._num_pts_per_orbit):  # type: ignore
                        x_cur = x[class_idx, :, i - 1, 0]
                        y_cur = x[class_idx, :, i - 1, 1]

                        if self.dynamical_system == 'pp_convention':
                            x[class_idx, :, i, 0] = (x_cur + p * y_cur * (1. - y_cur)) % 1
                            x[class_idx, :, i, 1] = (y_cur + p * x_cur * (1. - x_cur)) % 1
                        else:
                            x[class_idx, :, i, 0] = (x_cur + p * y_cur * (1. - y_cur)) % 1
                            x_next = x[class_idx, :, i, 0]
                            x[class_idx, :, i, 1] = (y_cur + p * x_next * (1. - x_next)) % 1


            self._orbits = x.reshape((-1, self._num_pts_per_orbit, 2))  #type: ignore

    def _orbit_high_precision(self, x_init, rho, num_points=1_000, precision=600):
        x_precise = np.zeros((1_000, 2))


        x0 = S(x_init[0])
        y0 = S(x_init[1])

        for i in range(0, num_points):
            x_precise[i, 0] = x0
            x_precise[i, 1] = y0

            x0 = N((x0 + rho * y0 * (1.0 - y0)) % 1, precision)
            y0 = N((y0 + rho * x0 * (1.0 - x0)) % 1, precision)

        return x_precise

    def _compute_persistence_diagrams(self) -> None:
        """ Computes the weak alpha persistence of the orbit data clouds.
            The result is stored in member variable ´persistence_diagrams´.
            It has the shape
            [num_classes, num_orbits, num_persistence_points, 3].
            In the last dimension the first two values are the coordinates of
            the points in the persistence diagrams and the third is the
            homology dimension.
        """
        if self._orbits is None:
            self._generate_orbits()
        
        # compute weak alpha persistence
        wap = WeakAlphaPersistence(
                            homology_dimensions=self._homology_dimensions,
                            n_jobs=self._n_jobs
                            )
        # c: class, o: orbit, p: point, d: dimension
        #orbits_stack = rearrange(self._orbits, 'c o p d -> (c o) p d')  # stack classes
        persistence_diagrams_categorical = wap.fit_transform(self._orbits)
        # shape: (num_classes * n_samples, n_features, 3)
        # Convert persistence diagram to one-hot homological dimension encoding
        self._persistence_diagrams = self._persistence_diagrams_to_one_hot(
                                        persistence_diagrams_categorical
                                        ).astype(self._dtype)

    def _persistence_diagrams_to_one_hot(self, persistence_diagrams):
        """ Convert homology dimension to one-hot encoding
        Args:
            persistence_diagrams ([np.array]): 
                persistence diagram with categorical
                homology dimension.
        Returns:
            [np.array]: 
                persistent diagram with one-hot encoded homology dimension.
        """
        if self._num_homology_dimensions > 1:
            return np.concatenate(
                (
                    persistence_diagrams[:, :, :2],  # point coordinates
                    (np.eye(self._num_homology_dimensions)  # type: ignore
                    [persistence_diagrams[:, :, -1].astype(np.int32)]),
                ),
                axis=-1)
        else:
            return persistence_diagrams[:, :, :2]
    
    def get_orbits(self) -> Union[None, np.ndarray]:
        """Returns the orbits as an ndarrays of shape
        (num_classes * num_orbits_per_class, num_pts_per_orbit, 2)
        Returns:
            np.ndarray: 
                Orbits
        """
        if self._orbits is None:
            self._generate_orbits()
        return self._orbits
    
    def get_persistence_diagrams(self) -> Union[None, np.ndarray]:
        """Returns the orbits as an ndarrays of shape
        (num_classes * num_orbits_per_class, num_topological_features, 3)
        Returns:
            np.ndarray: 
                Persistence diagrams
        """
        if self._persistence_diagrams is None:
            self._compute_persistence_diagrams()
            
        return self._persistence_diagrams
    
    def _split_data_idcs(self) -> None:
        """Split the data indices into training, validation and testing data.
        """
        idcs = np.arange(self._num_classes
                            * self._num_orbits_per_class)

        if self._test_percentage > 0.0:
            rest_idcs, self._test_idcs = train_test_split(idcs,
                                                        test_size=self._test_percentage)
        else:
            rest_idcs, self._test_idcs = idcs, []  # type: ignore
            
        if self._validation_percentage > 0.0:
            self._train_idcs, self._val_idcs = train_test_split(rest_idcs,
                                                    test_size =(
                                                    self._validation_percentage
                                                    / (1.0 - self._test_percentage))
                                                    )
        else:
            self._train_idcs, self._val_idcs = rest_idcs, []  # type: ignore
    
    def _get_data_loaders(self, list_of_arrays: List[np.ndarray],
                          dataloaders_kwargs: DataLoaderKwargs
                          ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Generates a DataLoader for the given list of arrays.
        Args:
            list_of_arrays ([List[np.ndarray]]): 
                List of arrays to load.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: 
                train, val, test data loaders.
        """
        assert ((self._train_idcs is not None) and 
                (self._val_idcs is not None) and
                (self._test_idcs is not None)),\
                "Train, validation, and test data must be initialized"
        
        dl_train = DataLoader(TensorDataset(*(torch.tensor(a[self._train_idcs])
                                              for a in list_of_arrays)),
                                **dataloaders_kwargs.get_train_args())
        dl_val = DataLoader(TensorDataset(*(torch.tensor(a[self._val_idcs])
                                            for a in list_of_arrays)),
                                **dataloaders_kwargs.get_val_args())
        dl_test = DataLoader(TensorDataset(*(torch.tensor(a[self._train_idcs])
                                             for a in list_of_arrays)),
                                **dataloaders_kwargs.get_test_args())
        return dl_train, dl_val, dl_test
    
    def get_dataloader_orbits(self,
                              dataloaders_kwargs: DataLoaderKwargs
                              )-> Tuple[DataLoader, DataLoader, DataLoader]:
        """Generates a Dataloader from the orbits dataset 
        Returns:
            DataLoader: 
                Dataloader of orbits
        """
        if self._orbits is None:
            self._generate_orbits()
        if self._train_idcs is None:
            self._split_data_idcs()
        return self._get_data_loaders([self._orbits.astype(self._dtype),  # type: ignore
                                            self._labels], # type: ignore
                                        dataloaders_kwargs)
    
    def get_dataloader_persistence_diagrams(self, dataloaders_kwargs: DataLoaderKwargs
                              )-> Tuple[DataLoader, DataLoader, DataLoader]:
        """Generates a Dataloader from the persistence diagrams dataset 
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]:
                Dataloaders of persistence diagrams
        """
        if self._persistence_diagrams is None:
            self._compute_persistence_diagrams()
        if self._train_idcs is None:
            self._split_data_idcs()

        return self._get_data_loaders([self._persistence_diagrams, self._labels], # type: ignore
                                        dataloaders_kwargs)
    
    def get_dataloader_combined(self, dataloaders_kwargs: DataLoaderKwargs,
                              )-> Tuple[DataLoader, DataLoader, DataLoader]:
        """Generates a Dataloader from the orbits dataset and the persistence diagrams
        Returns:
            DataLoader: 
                Dataloader of orbits and persistence diagrams
        """
        if self._persistence_diagrams is None:
            self._compute_persistence_diagrams()
        if self._train_idcs is None:
            self._split_data_idcs()
        return self._get_data_loaders([self._orbits,  # type: ignore
                                       self._persistence_diagrams,  # type: ignore
                                       self._labels],  # type: ignore
                                       dataloaders_kwargs)


def generate_orbit_parallel(
    num_classes,
    num_orbits,
    num_pts_per_orbit: int = 100,
    parameters: List[float] = [1.0],
    ) -> np.ndarray:
    """Generate sequence of points of a dynamical system
    in a parallel manner.

    Args:
        num_classes (int): 
            number of classes of dynamical systems.
        num_orbits (int): 
            number of orbits of dynamical system per class.
        num_pts_per_orbit (int, optional): 
            Number of points to generate.
            Defaults to 100.
        parameter (List[float], optional): 
            List of parameters of the dynamical
            system.
            Defaults to [1.0].

    Returns:
        np.ndarray: 
            Array of sampled points of the dynamical system.
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
        orbits (np.array): 
            Orbits of shape [n_points, 2]
        homology_dimensions (tuple, optional): 
            Dimensions to compute the
            persistence diagrams.
            Defaults to (0, 1).
        n_jobs (int, optional): 
            Number of cpus to use for parallel computation.
            Defaults to multiprocessing.cpu_count().

    Returns:
        np.array: 
            Array of persistence diagrams of shape
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
        diagrams (np.ndarray): 
            [description]
        num_classes (int): 
            [description]
        num_orbits (int): 
            [description]
        num_homology_dimensions (int, optional): 
            [description]. Defaults to 2.

    Returns:
        torch.tensor: 
            tuple of
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