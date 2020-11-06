import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from gdeep.neural_nets.utility import get_activations


def plot_PCA_activations(model, X_tensor, colors, n_components=2):
    """Plot PCA of the activations of all layers of the neural network

    Args:
        model ([type]): [description]
        X_tensor ([type]): [description]
        y_tensor ([type]): [description]
        n_components (int, optional): [description]. Defaults to 2.
    """
    activations_layers = get_activations(model,X_tensor)

    for i, activations_layer in enumerate(activations_layers.get_outputs()):
        pca = PCA(n_components=n_components)

        X_pca = pca.fit_transform(activations_layer)

        #TODO: fix this manual defintion
        color_scheme = colors

        plt.scatter(X_pca[:,0], X_pca[:,1],c=color_scheme)
        plt.title("hidden layer: " +str(i))
        plt.axis('equal')
        plt.show()
