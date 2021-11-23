# Test persistence_diagrams_to_sequence with MUTAG dataset

from os.path import join

import torch
import h5py  # type: ignore
import random

from gdeep.topology_layers import load_data
from gdeep.topology_layers.preprocessing\
    import persistence_diagrams_to_sequence, diagram_to_tensor


# Test persistence_diagrams_to_sequence

# def random_compatibility_test(n_trials: int = 10) -> None:
#     """Randomly check if persistence diagram is correctly converted to
#     sequence.

#     Raises:
#         Exception: Assertion error.

#     Returns:
#         None
#     """
#     filename = join("..", "..", "..", "examples", "graph_data",
#                     "MUTAG",
#                     "MUTAG" + ".hdf5")
#     diagrams_file = h5py.File(filename, "r")

#     # load_data with `load_data` methode
#     seq_pd, _, _ = load_data("MUTAG")

#     for _ in range(n_trials):

#         type_ = random.choice(list(diagrams_file.keys()))

#         type_index = list(diagrams_file.keys()).index(type_)

#         graph_number = random.choice(list(diagrams_file[type_].keys()))

#         # find indices that belong to type_
#         idx = seq_pd[int(graph_number)][:, 2 + type_index] == 1.0

#         computed_pts = seq_pd[int(graph_number)][idx][:, :2]

#         original_pts = torch.tensor(diagrams_file[type_][graph_number])

#         try:
#             assert torch.allclose(original_pts, computed_pts)
#         except AssertionError:
#             raise AssertionError("persistence_diagrams_to_sequence does not" +
#                                  "return the right sequence tensor")





def test_persistence_diagrams_to_sequence():
    tensor_dict = {"type1": {
                    "1": torch.tensor([[0.0066, 0.7961],
                                       [0.6612, 0.0359],
                                       [0.8394, 0.1597]]),
                    "2": torch.tensor([[0.1787, 0.1809],
                                       [0.2645, 0.5766],
                                       [0.5666, 0.1630],
                                       [0.9986, 0.0259]]),
                    "0": torch.tensor([[0.6910, 0.1265],
                                       [0.9085, 0.0230],
                                       [0.4977, 0.6386],
                                       [0.1331, 0.8196],
                                       [0.6929, 0.1859],
                                       [0.4216, 0.2283],
                                       [0.4996, 0.3380]]),
                    },
               "type2": {
                    "1": torch.tensor([[0.0932, 0.7327],
                                       [0.7248, 0.7940],
                                       [0.5550, 0.9960]]),
                    "2": torch.tensor([[0.9541, 0.6892],
                                       [0.7984, 0.8061],
                                       [0.5266, 0.0644],
                                       [0.0630, 0.2176]]),
                    "0": torch.tensor([[0.0896, 0.9181],
                                       [0.8755, 0.4239],
                                       [0.3665, 0.5990],
                                       [0.0960, 0.3615],
                                       [0.7895, 0.0670],
                                       [0.3407, 0.6902],
                                       [0.4052, 0.3058],
                                       [0.4820, 0.6540],
                                       [0.9083, 0.2075],
                                       [0.2015, 0.3533]])
                    }
               }

    output = persistence_diagrams_to_sequence(tensor_dict)

    expected_output = {1: torch.tensor(
            [[0.0066, 0.7961, 1.0000, 0.0000],
             [0.6612, 0.0359, 1.0000, 0.0000],
             [0.8394, 0.1597, 1.0000, 0.0000],
             [0.0932, 0.7327, 0.0000, 1.0000],
             [0.7248, 0.7940, 0.0000, 1.0000],
             [0.5550, 0.9960, 0.0000, 1.0000]]),
     2: torch.tensor(
            [[0.1787, 0.1809, 1.0000, 0.0000],
             [0.2645, 0.5766, 1.0000, 0.0000],
             [0.5666, 0.1630, 1.0000, 0.0000],
             [0.9986, 0.0259, 1.0000, 0.0000],
             [0.9541, 0.6892, 0.0000, 1.0000],
             [0.7984, 0.8061, 0.0000, 1.0000],
             [0.5266, 0.0644, 0.0000, 1.0000],
             [0.0630, 0.2176, 0.0000, 1.0000]]),
     0: torch.tensor(
            [[0.6910, 0.1265, 1.0000, 0.0000],
             [0.9085, 0.0230, 1.0000, 0.0000],
             [0.4977, 0.6386, 1.0000, 0.0000],
             [0.1331, 0.8196, 1.0000, 0.0000],
             [0.6929, 0.1859, 1.0000, 0.0000],
             [0.4216, 0.2283, 1.0000, 0.0000],
             [0.4996, 0.3380, 1.0000, 0.0000],
             [0.0896, 0.9181, 0.0000, 1.0000],
             [0.8755, 0.4239, 0.0000, 1.0000],
             [0.3665, 0.5990, 0.0000, 1.0000],
             [0.0960, 0.3615, 0.0000, 1.0000],
             [0.7895, 0.0670, 0.0000, 1.0000],
             [0.3407, 0.6902, 0.0000, 1.0000],
             [0.4052, 0.3058, 0.0000, 1.0000],
             [0.4820, 0.6540, 0.0000, 1.0000],
             [0.9083, 0.2075, 0.0000, 1.0000],
             [0.2015, 0.3533, 0.0000, 1.0000]]
     )
    }

    for i in range(3):
        try:
            assert(torch.allclose(output[i], expected_output[i]))
        except AssertionError:
            print("expected:\n", expected_output[i])
            print("actual:\n", output[i])
            raise AssertionError("persistence_diagrams_to_sequence" +
                                 "does not match")


# check if tensorised diagrams have the correct shape

def test_diagram_to_tensor():
    tensor_dict = {"type1": {
                    "1": torch.tensor([[0.0066, 0.7961],
                                       [0.6612, 0.0359],
                                       [0.8394, 0.1597]]),
                    "2": torch.tensor([[0.1787, 0.1809],
                                       [0.2645, 0.5766],
                                       [0.5666, 0.1630],
                                       [0.9986, 0.0259]]),
                    "0": torch.tensor([[0.6910, 0.1265],
                                       [0.9085, 0.0230],
                                       [0.4977, 0.6386],
                                       [0.1331, 0.8196],
                                       [0.6929, 0.1859],
                                       [0.4216, 0.2283],
                                       [0.4996, 0.3380]]),
                    },
               "type2": {
                    "1": torch.tensor([[0.0932, 0.7327],
                                       [0.7248, 0.7940],
                                       [0.5550, 0.9960]]),
                    "2": torch.tensor([[0.9541, 0.6892],
                                       [0.7984, 0.8061],
                                       [0.5266, 0.0644],
                                       [0.0630, 0.2176]]),
                    "0": torch.tensor([[0.0896, 0.9181],
                                       [0.8755, 0.4239],
                                       [0.3665, 0.5990],
                                       [0.0960, 0.3615],
                                       [0.7895, 0.0670],
                                       [0.3407, 0.6902],
                                       [0.4052, 0.3058],
                                       [0.4820, 0.6540],
                                       [0.9083, 0.2075],
                                       [0.2015, 0.3533]])
                    }
               }
    try:
        assert all((
                    diagram_to_tensor(
                        tensor_dict["type1"]).shape == torch.Size([3, 7, 2]),
                    diagram_to_tensor(
                        tensor_dict["type2"]).shape == torch.Size([3, 10, 2])
                ))
    except AssertionError:
        print("Converted diagrams do not have correct shape.")
        raise
