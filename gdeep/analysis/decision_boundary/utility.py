import numpy as np


class UniformlySampledPoint:
    """Generates n_samples uniformely random points in a box
    specified in tuple_list

    Args:
        tuple_list (list): list of dimensionwise upper and lower bounds of box
        n_samples (int, optional): number of sample points. Defaults to 1000.
    """

    def __init__(self, tuple_list: list, n_samples: int = 1000):

        self._dim = len(tuple_list)
        try:
            for (left, right) in tuple_list:
                assert left <= right
        except AssertionError:
            raise ValueError("Tuples have have to be non-empty intervals")

        scale = np.array([[right - left for (left, right) in tuple_list]])
        translation = np.array([[left for (left, _) in tuple_list]])

        self._sample_points = np.random.rand(n_samples, self._dim) * scale + translation

    def __call__(self):
        return self._sample_points

    def get_dim(self):
        """Returns dimension of sample point cloud

        Returns:
            int: dimension of point cloud
        """
        return self._dim
