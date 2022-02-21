from ..parallel_orbit import OrbitsGenerator
import numpy as np

parameters = (1.0, 2.0, 3.0)
num_orbits_per_class = 10
num_pts_per_orbit = 15

og = OrbitsGenerator(
    parameters=parameters,
    num_orbits_per_class=num_orbits_per_class,
    num_pts_per_orbit=num_pts_per_orbit
)

orbits = og.get_orbits()

def test_size_of_orbits():
    """Tests if the size of the generator is correct
    """
    
    
    assert orbits.shape == (len(parameters) * num_orbits_per_class, num_pts_per_orbit, 2)
    
def test_size_of_persistence_diagrams():
    """Test if the size of the persistence diagram is correct
    """
    
    persistence_diagrams = og.get_persistence_diagrams()
    
    assert [persistence_diagrams.shape[i] for i in (0, 2)]\
        == [len(parameters) * num_orbits_per_class, 2 + og._num_homology_dimensions]

def test_recursive_definition():
    """Tests if the recursively defined orbits are correct
    """
    for p_idx, p in enumerate(parameters):
        for o in range(num_orbits_per_class):
            for i in range(1, num_pts_per_orbit - 1):
                x_current = orbits[p_idx * num_orbits_per_class + o, i    , 0]
                x_next    = orbits[p_idx * num_orbits_per_class + o, i + 1, 0]
                y_current = orbits[p_idx * num_orbits_per_class + o, i    , 1]
                y_next    = orbits[p_idx * num_orbits_per_class + o, i + 1, 1]
                
                assert np.isclose(x_next, (x_current + p * y_current * (1.0 - y_current)) % 1.0)
                assert np.isclose(y_next, (y_current + p * x_next * (1.0 - x_next)) % 1.0)