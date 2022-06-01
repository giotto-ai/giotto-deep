import os

import numpy as np
from gdeep.data.datasets.persistence_diagrams_from_graphs_builder import PersistenceDiagramFromGraphBuilder
from gdeep.data.persistence_diagrams.one_hot_persistence_diagram import OneHotEncodedPersistenceDiagram,\
    get_one_hot_encoded_persistence_diagram_from_gudhi_extended
from gdeep.utility import DEFAULT_GRAPH_DIR



def test_mutag_load_and_save():

    # Parameters
    name_graph_dataset: str = 'MUTAG'
    diffusion_parameter: float = 0.1
    num_homology_types: int = 4


    # Create the persistence diagram dataset
    pd_creator = PersistenceDiagramFromGraphBuilder(name_graph_dataset, diffusion_parameter)
    pd_creator.create()

    gudhi_ext = \
    (np.array([[0.90634646, 0.90835863],
            [0.90659781, 0.90835863],
            [0.90672356, 0.90684934],
            [0.90672356, 0.90684934]]),
    np.array([[0.90634646, 0.90835863]]),
    np.array([[0.90659795, 0.90710085],
            [0.90672356, 0.90710077],
            [0.90672356, 0.90684934]]),
    np.array([[0.90659781, 0.90710077],
            [0.90659781, 0.90710085]]))

    pd_gudhi = get_one_hot_encoded_persistence_diagram_from_gudhi_extended(gudhi_ext)

    # Print sample extended persistence diagram
    file_path: str = os.path.join(DEFAULT_GRAPH_DIR,
                                f"MUTAG_{diffusion_parameter}_extended_persistence", "diagrams")
    graph_idx = 1
    pd = OneHotEncodedPersistenceDiagram.load(os.path.join(file_path, 
                                                        f"{graph_idx}.npy"))
    assert pd_gudhi.all_close(pd, atol=1e-6), \
        "Generated persistence diagram is not equal to the one from GUDHI"