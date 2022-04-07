from gdeep.topology_layers.preprocessing import \
    convert_gudhi_extended_persistence_to_persformer_input

import os

import numpy as np

from gdeep.utility.utils import ROOT_DIR

def test_convert_gudhi_extended_persistence_to_persformer_input() -> None:
    """Test the function 
    convert_gudhi_extended_persistence_to_persformer_input.
    
    Returns:
        None
    """
    
    # Load sample data
    path_to_data = os.path.join(ROOT_DIR, "gdeep", "topology_layers",
                                "tests", "data")

    # load lists in diagrams.pkl and labels.pkl
    with open(os.path.join(path_to_data, "diagrams.pkl"), "rb") as f:
        diagrams = np.load(f, allow_pickle=True)
        # diagrams has the type List[List[Tuple[int, Tuple[float, float]]]]
        # where the first element is the index of the diagram, the second element
        # are the four different extended persistence types, and the third element
        # is a list of pairs (homology_dim, (birth, death))
    with open(os.path.join(path_to_data, "labels.pkl"), "rb") as f:
        labels = np.load(f, allow_pickle=True)  # List[int] of size 5
        
    input_size = len(diagrams)

    assert len(diagrams) == len(labels),\
        "The number of diagrams and labels must be the same"
    
    encoded_diagrams = \
    convert_gudhi_extended_persistence_to_persformer_input(diagrams)
    
    assert encoded_diagrams.shape == (input_size, 5577, 6),\
        "The shape of the encoded diagrams is incorrect"
