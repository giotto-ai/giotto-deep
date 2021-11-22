# %%
import numpy as np
import matplotlib.pyplot as plt
# %%

def get_orbits(
        x_init,
        parameters = (2.5, 3.5, 4.0, 4.1, 4.3),
        num_classes = 5,
        num_orbits_per_class = 5,
        num_pts_per_orbit = 1_000,
    ):

    # Initialize the orbits array with zeros.
    x = np.zeros((
        num_classes,  # type: ignore
        num_orbits_per_class,
        num_pts_per_orbit,
        2
    ))

    # Initialize the labels array with the hyperparameter indices.
    # y = np.array([num_orbits_per_class * [c]
    #                for c in range(num_classes)])

    #labels = y.reshape(-1)

    # generate dataset
    for class_idx, p in enumerate(parameters):  # type: ignore
        x[class_idx, :, 0, :] = x_init[class_idx]  # type: ignore

        for i in range(1, num_pts_per_orbit):  # type: ignore
            x_cur = x[class_idx, :, i - 1, 0]
            y_cur = x[class_idx, :, i - 1, 1]


            x[class_idx, :, i, 0] = (x_cur + p * y_cur * (1. - y_cur)) % 1
            x_next = x[class_idx, :, i, 0]
            x[class_idx, :, i, 1] = (y_cur + p * x_next * (1. - x_next)) % 1
                
    orbits = x.reshape((-1, num_pts_per_orbit, 2))#.astype('f')
    return orbits
# %%
num_orbits_per_class = 5
num_classes = 5
x_init = np.random.rand(num_classes, num_orbits_per_class, 2)

x_32  = get_orbits(x_init.astype('float32'))
x_64  = get_orbits(x_init.astype('float64'))
x_128 = get_orbits(x_init.astype('float128'))
# %%
plt.scatter(x_32[0, :, 0], x_32[0, :, 1])
# %%
plt.scatter(x_128[0, :, 0], x_128[0, :, 1])
# %%
def distance_torus(x, y):
    x_torus = np.stack([x,
                        1.0 - x,
                        np.array([[1.0, 0.0]]) - x,
                        np.array([[1.0, 0.0]]) - x], axis=0)
    distances = np.sqrt(((x_torus - np.expand_dims(y, axis=0))**2).sum(axis=-1)).min(axis=0)


    return distances
plt.yscale('log')
plt.plot(distance_torus(x_64[0], x_128[0]))
# %%

# %%
plt.scatter(x_precise[:, 0], x_precise[:, 1])
# %%
plt.scatter(x_32[0, :, 0], x_32[0, :, 1])
# %%
plt.plot(distance_torus(x_precise_old, x_precise))
# %%
x_precise_old = x_precise
# %%

from sympy import N, S

def orbit_high_precision(x_init, rho, num_points=1_000, precision=600):
    x_precise = np.zeros((1_000, 2))


    x0 = S(x_init[0])
    y0 = S(x_init[1])

    for i in range(0, num_points):
        x_precise[i, 0] = x0
        x_precise[i, 1] = y0

        x0 = N((x0 + rho * y0 * (1.0 - y0)) % 1, precision)
        y0 = N((y0 + rho * x0 * (1.0 - x0)) % 1, precision)

    return x_precise

def get_orbits(
        parameters = (2.5, 3.5, 4.0, 4.1, 4.3),
        num_classes = 5,
        num_orbits_per_class = 5,
        num_pts_per_orbit=1_000,
    ):

    x = np.zeros((
        num_classes,  # type: ignore
        num_orbits_per_class,
        num_pts_per_orbit,
        2
    ))

    # Initialize the labels array with the hyperparameter indices.
    #y = np.array([num_orbits_per_class * [c]
    #               for c in range(num_classes)])

    #labels = y.reshape(-1)

    # generate dataset
    for class_idx, p in enumerate(parameters):  # type: ignore
        x[class_idx, :, 0, :] = np.random.rand(self._num_orbits_per_class, 2)  # type: ignore

        for orbit in range(num_orbits_per_class):
            x[class_idx, orbit, :, :] = orbit_high_precision(
                                            x[class_idx, orbit, 0, :],
                                            p,
                                            precision=600)
                
    orbits = x.reshape((-1, num_pts_per_orbit, 2))#.astype('f')
    return orbits
# %%
x_32 = orbit_high_precision(np.array([0.000001, 0.000001]), 2.5, precision=300)
x = orbit_high_precision(np.array([0.000001, 0.000001]), 2.5, precision=600)

plt.yscale('log')
plt.plot(distance_torus(x_32, x))
# %%
