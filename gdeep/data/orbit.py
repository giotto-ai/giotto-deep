import numpy as np  # type: ignore
import warnings


def generate_orbit(num_pts_per_orbit: int = 100,
                   parameter: float = 1.0) -> np.ndarray:
    """Generate sequence of points of a dynamical system
    Non-parallel version

    Args:
        num_pts_per_orbit (int, optional): 
            Number of points to generate.
            Defaults to 100.
        parameter (float, optional): 
            Parameter of the dynamical system.
            Defaults to 1.0.

    Returns:
        np.ndarray: 
            Array of sampled points of the dynamical system.
    """
    warnings.warn("This methode is deprecated. Use the parallel\
                   version create_data.parallel_orbit.generate_\
                       orbit_parallel",
                  DeprecationWarning)
    try:
        assert parameter > 0
    except AssertionError:
        print('Parameter must be greater than 0')
    try:
        assert num_pts_per_orbit > 0
    except AssertionError:
        print('num_pts_per_orbit must be greater than 0')

    orbit = np.zeros([num_pts_per_orbit, 2])
    x_cur, y_cur = np.random.rand(), np.random.rand()  # current points
    for idx in range(num_pts_per_orbit):
        x_cur = (x_cur + parameter * y_cur * (1. - y_cur)) % 1
        y_cur = (y_cur + parameter * x_cur * (1. - x_cur)) % 1
        orbit[idx, :] = [x_cur, y_cur]
    return orbit
