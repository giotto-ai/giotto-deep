import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def gradient_flow(neural_net: nn.Module,  boundary_tuple: list,\
    n_samples: int = 3000,\
    epsilon: float = 0.01, n_epochs: int = 5000):
    """ Computes a the decision boundary of a neural network using
    gradient flow

    Args:
        neural_net (nn.Module): neural network trained on a binary
                                classification task
        boundary_tuple (list): [description]
        n_samples (int, optional): [description]. Defaults to 1000.
        n_epochs (int, optional): [description]. Defaults to 1000.

    Returns:
        numpy.ndarray: row wise sample of points on the decision boundary
    """

    sample_points = sample_points_uniformly(boundary_tuple, n_samples=n_samples)



    sample_points_tensor = torch.from_numpy(sample_points).float()

    for _ in range(0,n_epochs):

        delta = torch.zeros_like(sample_points_tensor, requires_grad=True)

        predict = neural_net.forward(x_cat=None, x_cont=sample_points_tensor + delta)
        
        predict_first_class = predict[:,0]
        predict_second_class = predict[:,1]

        loss = torch.sum((predict_first_class-0.5)**2+(predict_second_class-0.5)**2)
        loss.backward()

        sample_points_tensor -= epsilon * delta.grad.detach()

    delta = torch.zeros_like(sample_points_tensor, requires_grad=True)

    predict = neural_net.forward(x_cat=None, x_cont=sample_points_tensor+delta)
    
    predict_first_class = predict[:,0]
    predict_second_class = predict[:,1]

    

    loss = torch.sum((predict_first_class-0.5)**2+(predict_second_class-0.5)**2)

    loss.backward()


    sample_points_tensor = sample_points_tensor[\
        torch.stack((
        #torch.norm(delta.grad.detach(), dim=1)> 1.e-05,\
        (1.-predict_first_class>1e-1)[:],\
        (predict_first_class>1e-1)[:]\
        ),dim=1).all(dim=1)\
        ]

    sample_points_boundary = sample_points_tensor.numpy()

    # print(predict[\
    #     torch.stack((
    #     #torch.norm(delta.grad.detach(), dim=1)> 1.e-05,\
    #     (1.-predict_first_class>1e-1)[:],\
    #     (predict_first_class>1e-1)[:]\
    #     ),dim=1).all(dim=1)\
    #     ].detach().numpy())

    return sample_points_boundary


def sample_points_uniformly(tuple_list: list, n_samples: int=1000):
    """ Sample uniformaly random in a box

    Args:
        tuple_list (list): list of intervals
        n_samples (int): number of sample points
    """
    dim = len(tuple_list)
    try:
        for (left, right) in tuple_list:
            assert(left <= right)
    except:
        print("Tuples have have to be non-empty intervals")

    scale = np.array([[right-left for (left, right) in tuple_list]])
    translation = np.array([[left for (left, _) in tuple_list]])

    sample_point = np.random.rand(n_samples, dim) * scale + translation

    return sample_point