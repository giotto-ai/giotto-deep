# %%

import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from sklearn import svm
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score

from persim import PersImage, sliced_wasserstein
from persim import PersistenceImager
from gtda.homology import VietorisRipsPersistence


from sklearn.model_selection import train_test_split
# %%
# Create ORBIT5K dataset
parameters = (2.5, 3.5, 4.0, 4.1, 4.3)  # different classes of orbits
homology_dimensions = (0)

config = {
    'parameters': parameters,
    'num_classes': len(parameters),
    'num_orbits': 100,  # number of orbits per class
    'num_pts_per_orbit': 1_000,
    'homology_dimensions': homology_dimensions,
    'num_homology_dimensions': len(homology_dimensions),
    'validation_percentage': 100,  # size of validation dataset relative
    # to training
    'dynamical_system': 'pp_convention',  # either use persistence paths
    # convention ´pp_convention´ or the classical convention
    # ´classical_convention´
}

x = np.zeros((
                config['num_classes'],  # type: ignore
                config['num_orbits'],
                config['num_pts_per_orbit'],
                2
            ))

y = np.array([config['num_orbits'] * [c] for c in range(config['num_classes'])])
 
y = y.reshape(-1)

# generate dataset
for cidx, p in enumerate(config['parameters']):  # type: ignore
    x[cidx, :, 0, :] = np.random.rand(config['num_orbits'], 2)  # type: ignore

    for i in range(1, config['num_pts_per_orbit']):  # type: ignore
        x_cur = x[cidx, :, i - 1, 0]
        y_cur = x[cidx, :, i - 1, 1]

        if config['dynamical_system'] == 'pp_convention':
            x[cidx, :, i, 0] = (x_cur + p * y_cur * (1. - y_cur)) % 1
            x[cidx, :, i, 1] = (y_cur + p * x_cur * (1. - x_cur)) % 1
        else:
            x[cidx, :, i, 0] = (x_cur + p * y_cur * (1. - y_cur)) % 1
            x_next = x[cidx, :, i, 0]
            x[cidx, :, i, 1] = (y_cur + p * x_next * (1. - x_next)) % 1
            
x = x.reshape((-1, config['num_pts_per_orbit'], 2))

# %%
# Use sliced wasserstein kernel and random forest algorithm for the
# classification

VR = VietorisRipsPersistence(homology_dimensions=config['homology_dimensions'])


diagrams = VR.fit_transform(x)

    
# %%

test_size = config['validation_percentage'] / \
    (100.0 + config['validation_percentage'])

X_train, X_test, y_train, y_test = train_test_split(diagrams[:, : , :2],
                                                    y,
                                                    test_size=test_size,
                                                    random_state=42)
# %%
# Only use 30 train and 30 validation datapoint because of huge
# computation time.
X_train = X_train[:30]
y_train = y_train[:30]
X_test = X_test[:30]
y_test = y_test[:30]
# %%
def gram_sw_matrix(X_1, X_2, n_iter=50, verbose=False):
    gram = np.zeros((X_1.shape[0], X_2.shape[0]))
    
    # non-parallel version
    # dm = pdist(X.reshape((X.shape[0], -1)),
    #                            lambda u, v: sliced_wasserstein(u.reshape((-1, 2)),
    #                                                            v.reshape((-1, 2)), n_iter))
    pairwise_distances(X_1.reshape((X_1.shape[0], -1)),
                   X_2.reshape((X_2.shape[0], -1)),
                   lambda u, v: sliced_wasserstein(u.reshape((-1, 2)),
                                                v.reshape((-1, 2)), 10),
                   n_jobs=5)
    counter = 0
    for i in range(X_1.shape[0]):
        for j in range(i + 1, X_2.shape[0]):
            sw = dm[counter]
            gram_train[i, j] = sw
            gram_train[j, i] = sw
    return gram

gram_train = gram_sw_matrix(X_train, X_train, verbose=True)
gram_test = gram_sw_matrix(X_train, X_test)

# %%
from sklearn.multiclass import OneVsOneClassifier
clf = svm.SVC(kernel='precomputed')
ovo = OneVsOneClassifier(clf)
ovo.fit(gram_train, y_train)

pred_train = ovo.predict(gram_train)
print("accuracy: %.2f%%" % (accuracy_score(pred_train, y_train)))

pred_test = ovo.predict(gram_test)

print("accuracy: %.2f%%" % (accuracy_score(pred_test, y_test)))
# %%
