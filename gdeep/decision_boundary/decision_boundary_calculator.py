# Joint with Matthias Kemper

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable


class DecisionBoundaryCalculator():
    """
    Abstract class for calculating the decision boundary of
    a neural network
    """
    def step(self):
        """ Performs a single step towards the decision boundary
        """
        pass

    def return_decision_boundary(self)->torch.tensor:
        """Return current state and does not make a step

        Returns:
            torch.tensor: current estimate of the decision boundary
        """
        pass
        



class GradientFlowDecisionBoundaryCalculator(DecisionBoundaryCalculator):
    """
    Computes Decision Boundary using the gradient flow method
    """
    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor],
                       initial_points: torch.Tensor,
                       optimizer: Callable[[torch.Tensor], torch.optim.Optimizer]):
        """Transforms `model` to `self.model` of output shape (N) and initizalizes `self.optimizer`.
        Args:
            model (Callable[[torch.Tensor], torch.Tensor]): Function that maps a `torch.Tensor` of shape 
                (N, D_in) to a tensor either of shape (N) and with values in [0,1] or of shape (N, D_out)
                with values in [0,1] such that the last axis sums to 1.
            initial_points (torch.tensor): `torch.Tensor` of shape (N, D_in)
            optimizer (Callable[[torch.Tensor], torch.optim.Optimizer]): Function returning an optimizer
                for the params given as an argument
        """
        self.sample_points = initial_points
        self.sample_points.requires_grad = True
        
        output = model(self.sample_points)
        output_shape = output.size()

        
        if not len(output_shape) in [1, 2]:
            raise RuntimeError('Output of model has wrong size!')

        if len(output_shape) == 1:
            self.model = lambda x: (model(x) - 0.5)**2
        elif output_shape[-1] == 1:
            self.model = lambda x: (model(x).reshape((-1)) - 0.5)**2
        elif output_shape[-1] == 2:
            def new_model(x):
                y = model(x)
                return (y[:,0] - 0.5)**2
            self.model = new_model
        else:
            def new_model(x):
                y = torch.topk(x, 2).values
                return (y[:,0] - y[:,1])**2
            self.model = new_model
        #Check if self.model has the right output shape
        assert len(self.model(self.sample_points).size())==1, f'Output shape is {self.model(self.sample_points).size()}'

        self.optimizer = optimizer([self.sample_points])

    def step(self):
        """Performs a single step towards the decision boundary
        """

        self.optimizer.zero_grad()
        loss = torch.sum(self.model(self.sample_points))
        loss.backward()
        self.optimizer.step()

    
    def return_decision_boundary(self)->torch.tensor:
        return self.sample_points