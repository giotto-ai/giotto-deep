from typing import List, Tuple

import numpy as np

from gdeep.utility.utils import flatten_list_of_lists

def convert_gudhi_extended_persistence_to_persformer_input(
    diagrams: List[List[Tuple[int, Tuple[float, float]]]]) -> np.ndarray:
    """Convert the diagrams from Gudhi's extended persistence of graphs format
    to PersFormal's input format.
    
    Args:
        diagrams (List[List[Tuple[int, Tuple[float, float]]]]):
            The persistence diagrams in Gudhi's extended persistence format
            of a list of graphs.
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
    
    input_size = len(diagrams)
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

def _convert_single_graph_extended_persistence_to_one_hot_array(
    diagram: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Convert an extended persistence diagram of a single graph to an
    array with one-hot encoded homology type.

    Args:
        diagram (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
            The diagram of an extended persistence of a single graph.
    
    Returns:
        np.ndarray:
            The diagram in one-hot encoded homology type of size
            (num_points, 6).
    """
    # Get the length of each array
    lengths = [len(array) for array in diagram]
    
    if lengths == [0, 0, 0, 0]:
        return np.zeros((0, 6))
    
    # One-hot encode the homology type
    homology_type: np.ndarray = np.array(flatten_list_of_lists(
        [[i] * lengths[i] for i in range(4)]
        ))
    homology_type_one_hot = np.zeros((sum(lengths), 4))
    homology_type_one_hot[np.arange(sum(lengths)), homology_type] = 1
    
    # Concatenate the arrays
    diagram_one_hot = np.concatenate([sub_diagram for sub_diagram in diagram],
                                     axis=0)
    diagram_one_hot = np.concatenate([diagram_one_hot, homology_type_one_hot],
                                     axis=1)
    return diagram_one_hot