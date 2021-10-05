"""Testing for decision_boundary_calculator."""
# License: GNU AGPLv3
from .. import GradientFlowDecisionBoundaryCalculator
from ....data import *
from ....models import FFNet
import torch


def test_gfdbc_2_dim():

    circle_detect_nn = FFNet([2, 2])

    g = GradientFlowDecisionBoundaryCalculator(
            model=circle_detect_nn,
            initial_points=torch.rand((100, 2)),
            optimizer=lambda params: torch.optim.Adam(params)
    )
    for i in range(100):
        g.step()
    assert g.get_decision_boundary().size() == torch.Size([100, 2])


def test_gfdbc_multiclass():

    circle_nn_3d = FFNet([3, 3])

    g = GradientFlowDecisionBoundaryCalculator(model=
                                               circle_nn_3d,
                                               initial_points=
                                               torch.rand((100, 3)),
                                               optimizer=
                                               lambda params:
                                               torch.optim.Adam(params))
    for i in range(100):
        g.step()
    assert g.get_decision_boundary().size() == torch.Size([100, 3])

# TODO: Check if significant number of points lie close to the decision boundary
