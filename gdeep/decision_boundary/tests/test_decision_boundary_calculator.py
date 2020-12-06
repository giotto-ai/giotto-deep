"""Testing for compute_boundary."""
# License: GNU AGPLv3

import numpy as np

import pytest

import torch
import torch.nn as nn

from gdeep.decision_boundary.decision_boundary_calculator import *


def test_gfdbc_2_dim():
    class CircleNN(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.dim = 2

                    
        def forward(self, x_cont):
            try:
                assert(x_cont.shape[-1]==2)
            except:
                raise ValueError(f'input has to be a {2}-dimensional vector')
            activation = 0.5*torch.exp(-torch.sum(x_cont**2, axis=-1)+1)-0.5
            return activation.reshape((-1,1))
        
        def return_input_dim(self):
            return 2
    circle_detect_nn = CircleNN()

    g = GradientFlowDecisionBoundaryCalculator(
            model=circle_detect_nn,
            initial_points=torch.rand((100,2)),
            optimizer=lambda params: torch.optim.Adam(params)
    )
    for i in range(100):
        g.step()
    assert g.return_decision_boundary().size() == torch.Size([100, 2])

def test_gfdbc_multiclass():
    class CircleNN3D(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x_cont):
            activation = torch.exp(x_cont**2)
            return activation
    circle_nn_3d = CircleNN3D()

    g = GradientFlowDecisionBoundaryCalculator(
            model=circle_nn_3d,
            initial_points=torch.rand((100,3)),
            optimizer=lambda params: torch.optim.Adam(params)
    )
    for i in range(100):
        g.step()
    assert g.return_decision_boundary().size() == torch.Size([100, 3])