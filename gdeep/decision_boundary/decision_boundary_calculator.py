# Joint with Matthias Kemper

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable

from abc import ABC
from abc import abstractmethod

class DecisionBoundaryCalculator(ABC):
    """
    Abstract class for calculating the decision boundary of
    a neural network
    """
    @abstractmethod
    def step(self):
        """ Performs a single step towards the decision boundary
        """
        pass
    
    @abstractmethod
    def get_decision_boundary(self)->torch.tensor:
        """Return current state and does not make a step

        Returns:
            torch.tensor: current estimate of the decision boundary
        """
        pass
    
    def _convert_to_distance_to_boundary(self, model, output_shape):
        """Convert the binary or multiclass classifier to a scalar valued model with
        distance to the decision boundary as output.
        """
        if len(output_shape) == 1:
            new_model = lambda x: torch.abs(model(x) - 0.5)
        elif output_shape[-1] == 1:
            new_model = lambda x: torch.abs(model(x).reshape((-1)) - 0.5)
        elif output_shape[-1] == 2:
            new_model = lambda x: torch.abs(model(x)[:,0] - 0.5)
        else:
            def new_model(x):
                y = torch.topk(x, 2).values
                return y[:,0] - y[:,1]

        return new_model


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
        new_model = self._convert_to_distance_to_boundary(model, output_shape)
        self.model = lambda x: new_model(x)**2

        # Check if self.model has the right output shape
        assert len(self.model(self.sample_points).size())==1, f'Output shape is {self.model(self.sample_points).size()}'

        self.optimizer = optimizer([self.sample_points])

    def step(self):
        """Performs a single step towards the decision boundary
        """

        self.optimizer.zero_grad()
        loss = torch.sum(self.model(self.sample_points))
        loss.backward()
        self.optimizer.step()

    
    def get_decision_boundary(self) -> torch.Tensor:
        return self.sample_points

    def get_filtered_decision_boundary(self, delta = 0.25) -> torch.Tensor:
        q_dist = self.model(self.sample_points)
        return self.sample_points[q_dist<=delta]