"""Testing for compute_boundary."""
# License: GNU AGPLv3

import numpy as np
import pytest
import torch

from gdeep.decision_boundary import UniformlySampledPoint, PrintGradientFlow, GradientFlow

def test_sample_points_uniformly():
    sample_points = UniformlySampledPoint([(2,3), (1,2)], n_samples=1000)

    print(sample_points().shape)
    assert sample_points().shape == (1000,2)

    
    assert (sample_points().min(axis=0)>=np.array([2,1])).all()

    assert (sample_points().min(axis=0)<=np.array([3,2])).all()