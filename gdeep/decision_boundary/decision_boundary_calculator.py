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
        """[summary]

        Args:
            model (nn.Module): [description]
            initial_points (torch.tensor): [description]
            optimizer (Callable): Function returning an optimizer for the params given as an argument

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """
        self.sample_points = initial_points
        self.sample_points.requires_grad = True
        
        
        output_shape = model.forward(self.sample_points).shape
        assert(len(output_shape)==2)

        if output_shape[-1]==1:
            self.model = model
        elif output_shape[-1]==2:
            class AbsoluteDifference(nn.Module):
                def forward(self, x):
                    #x.shape = (*,2)
                    x = x.reshape((x.shape[0],1,x.shape[-1]))
                    return torch.abs(F.conv1d(x, weight=torch.tensor([[[1.,-1.]]])).reshape((-1)))
            self.model = nn.Sequential(model, absolute_difference)
        else:
            # TODO: Implement default multiclass distance to decision boundary
            # function
            raise NotImplementedError
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