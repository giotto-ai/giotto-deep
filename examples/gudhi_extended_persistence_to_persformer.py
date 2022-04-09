# %%

from copyreg import pickle
from gdeep.utility.utils import autoreload_if_notebook
from gdeep.utility.constants import ROOT_DIR

autoreload_if_notebook()

import os
from typing import List, Tuple

import numpy as np

# %%

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
    
# %%
def convert_gudhi_extended_persistence_to_persformer_input(
    diagrams: List[List[Tuple[int, Tuple[float, float]]]]) -> np.ndarray:
    """Convert the diagrams from Gudhi's extended persistence format to
    PersFormal's input format.
    
    Args:
        diagrams (List[List[Tuple[int, Tuple[float, float]]]]):
            The persistence diagrams in Gudhi's extended persistence format.
            See https://gudhi.inria.fr/python/latest/simplex_tree_ref.html#gudhi.SimplexTree.extended_persistence  # noqa
            
    Returns:
        np.ndarray:
            The diagrams in Persformers's input format. This is a numpy array
            with shape (num_diagrams, num_points, 2 + 4) where the first one
            is the index of the diagram, the second one is the index of the
            of the point, and the last one is the birth and death time of the
            point and the one-hot vector of the extended persistence type.
            The diagrams are padded with zeros to have the same length.
    """
    
    # For each diagram, one-hot encode the four different labels
    # and concatenate them to a single np.array for each diagram
    encoded_diagrams_list = []
    for k, diagram in enumerate(diagrams):
        # Flatten the list of extended persistence types
        diagram_flatten = [(i, item[1][0], item[1][1])  # type: ignore
            for i, sublist in enumerate(diagrams[0])
            for item in sublist]
        
        encoded_diagram = np.zeros((len(diagram_flatten), 6))
        for i, (label, birth, death) in enumerate(diagram_flatten):
            # put birth and death coordinates in the first and second columns
            encoded_diagram[i, 0] = birth
            encoded_diagram[i, 1] = death
            
            # One-hot encode the label
            encoded_diagram[i, 2 + label] = 1
            
        encoded_diagrams_list.append(encoded_diagram)
        
    # concatenate all diagrams into a single np.array of shape
    # (input_size, max_num_points, 6) by filling the remaining
    # entries with zeros
    max_num_points = max(len(diagram) for diagram in encoded_diagrams_list)
    encoded_diagrams = np.zeros((input_size, max_num_points, 6))
    for i, diagram in enumerate(encoded_diagrams_list):  # type: ignore
        encoded_diagrams[i, :len(diagram)] = encoded_diagrams_list[i]
    
    return encoded_diagrams
    
encoded_diagrams = \
    convert_gudhi_extended_persistence_to_persformer_input(diagrams)
    
assert encoded_diagrams.shape == (input_size, 5577, 6),\
    "The shape of the encoded diagrams is incorrect"



# %%
