import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class GradientFlow():
    """ Computes a the decision boundary of a neural network using
    gradient flow

    Args:
        neural_net (nn.Module): neural network trained on a binary
                                classification task
        boundary_tuple (list): [description]
        n_samples (int, optional): Number of sample points. Defaults to 1000.
        n_epochs (int, optional): Number of epochs for the gradient flow. Defaults to 30.
    """

    def __init__(self, neural_net: nn.Module,  boundary_tuple: list,\
        n_samples: int = 10000,\
        epsilon: float = 0.01, n_epochs: int = 30):

        self.neural_net = neural_net
        self.n_samples = n_samples
        self.epsilon = epsilon
        self.n_epochs = n_epochs
        self.sample_points = UniformlySampledPoint(boundary_tuple, n_samples=self.n_samples)
        self.sample_points_tensor = torch.from_numpy(self.sample_points()).float()

    def gradient(self):

        delta = torch.zeros_like(self.sample_points_tensor, requires_grad=True)

        prediction = self.neural_net.forward(None, self.sample_points_tensor + delta)

        loss = torch.sum((prediction-0.5)**2)

        loss.backward()

        return delta.grad.detach()

    def predict(self):
        return self.neural_net.forward(None, self.sample_points_tensor)

    
    def gradient_step(self):
        self.sample_points_tensor -= self.epsilon * self.gradient()

    

    def gradient_flow(self):
        for _ in range(self.n_epochs):
            self.gradient_step()


    def compute_boundary(self):
        self.gradient_flow()
        predict = self.predict()

        sample_points_db_tensor = self.sample_points_tensor[\
            torch.stack((
            (predict>4e-1)[:,0],\
            (predict>4e-1)[:,1]\
            ),dim=1).all(dim=1)\
            ]
        
        sample_points_db = sample_points_db_tensor.numpy()
        # in case the nn prediction was poor
        if len(sample_points_db) > 0:
            return sample_points_db
        return torch.zeros_like(self.sample_points_tensor).numpy()

    def __call__(self):
        return self.compute_boundary()


    def return_sample_points(self):
        return self.sample_points


class PrintGradientFlow():

    def __init__(self, show_gradient: bool = True):
        pass

class UniformlySampledPoint():
    """ Sample uniformly random in a box

    Args:
        tuple_list (list): list of intervals
        n_samples (int): number of sample points
    """
    def __init__(self, tuple_list: list, n_samples: int=1000):
        self._dim = len(tuple_list)
        try:
            for (left, right) in tuple_list:
                assert(left <= right)
        except:
            print("Tuples have have to be non-empty intervals")

        scale = np.array([[right-left for (left, right) in tuple_list]])
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
