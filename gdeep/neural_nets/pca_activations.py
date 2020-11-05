import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import utility


def plot_PCA_activations(model, X_tensor, y_tensor, n_components=2):
    """Plot PCA of the activations of all layers of the neural network

    Args:
        model ([type]): [description]
        X_tensor ([type]): [description]
        y_tensor ([type]): [description]
        n_components (int, optional): [description]. Defaults to 2.
    """
    activations_layers = utility.get_activations(model,X_tensor)

    for i, activations_layer in enumerate(activations_layers.get_outputs()):
        pca = PCA(n_components=n_components)

        X_pca = pca.fit_transform(activations_layer)

        plt.scatter(X_pca[:,0], X_pca[:,1],c=y_tensor)
        plt.title("hidden layer: " +str(i))
        plt.show()
