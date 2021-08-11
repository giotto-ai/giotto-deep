from typing import Tuple, Dict

from sklearn.preprocessing import LabelEncoder  # type: ignore
from os.path import join, isfile

import h5py  # type: ignore

import torch
from torch import Tensor
import torch.nn.functional as F

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def persistence_diagrams_to_sequence(
        tensor_dict: Dict[str, Dict[str, Tensor]]
     ):
    """Convert tensor dictionary to sequence of Tensors
        Output will be a List of tensors of the shape [graphs, absolute
        number of points per graph, 2(for the x and y coordinate)
        + number of types]

    Args:
        tensor_dict (Dict[str, Dict[str, Tensor]]): Dictionary of types and
            Dictionary of graphs and Tensors of points in the persistence
            diagrams.

    Returns:
        Dict[Int, Tensor]: List of tensors of the shape described above
    """
    types = list(tensor_dict.keys())

    sequence_dict = {}

    def encode_points(graph_idx, type_idx, type_, n_pts):
        one_hot = F.one_hot(
                torch.tensor([type_idx] * n_pts),
                num_classes=len(types))
        return torch.cat([
                    tensor_dict[type_][str(graph_idx)],
                    one_hot.expand((n_pts, len(types)))
                ], axis=-1)

    for graph_idx in [int(k) for k in tensor_dict[types[0]].keys()]:
        tensor_list = []
        for type_idx, type_ in enumerate(types):
            n_pts = tensor_dict[type_][str(graph_idx)].shape[0]
            if(n_pts > 0):
                tensor_list.append(encode_points(graph_idx,
                                                 type_idx,
                                                 type_,
                                                 n_pts))
        sequence_dict[graph_idx] = torch.cat(tensor_list,
                                             axis=0)  # type: ignore
    return sequence_dict


def load_data(
        dataset_: str = "MUTAG",
        path_dataset: str = "graph_data",
        verbose: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load dataset from files.

    Args:
        dataset (str, optional): File name of the dataset to load. There should
            be a hdf5 file for the extended persistence diagrams of the dataset
            as well as a csv file for the additional features in the path
            dataset directory. Defaults
            to "MUTAG".
        path_dataset (str, optional): Directory name of the dataset to load.
            Defaults to None.
        verbose (bool, optional): If `True` print size of the loaded dataset.
            Defaults to False.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of the loaded
            dataset consisting of the persistent features of the graphs, the
            additional features.
    """
    filenames = {}
    for file_suffix in [".hdf5", ".csv"]:
        try:
            filenames[file_suffix] = join(path_dataset,
                                          dataset_,
                                          dataset_ + file_suffix)
            assert(isfile(filenames[file_suffix]))
        except AssertionError:
            print(dataset_ + file_suffix +
                  " does not exist in given directory!")
    diagrams_file = h5py.File(filenames[".hdf5"], "r")
    # directory with persistance diagram type as keys
    # every directory corresponding to a key contains
    # subdirectories '0', '1', ... corresponding to the graphs.
    # For example, one can access a diagram by
    # diagrams_file['Ext1_10.0-hks']['1']
    # This is a hdf5 dataset object that contains the points of then
    # corresponding persistence diagram. These may contain different
    # numbers of points.

    # list of tensorised persistence diagrams

    additional_features = pd.read_csv(filenames[".csv"], index_col=0, header=0)
    labels = additional_features[['label']].values  # true labels of graphs
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels.reshape(-1))
    x_features = np.array(additional_features)[:, 1:]
    # additional graph features

    number_of_graphs = additional_features.shape[0]
    # number of graphs in the dataset

    # convert values in diagrams_file from numpy.ndarray to torch.tensor
    tensor_dict = {}  # type: ignore

    for type_ in diagrams_file.keys():
        tensor_dict[type_] = {}
        for graph in diagrams_file[type_].keys():
            tensor_dict[type_][graph] = torch.tensor(
                                            diagrams_file[type_][graph]
                                            )

    if verbose:
        print(
            "Dataset:", dataset_,
            "\nNumber of graphs:", number_of_graphs,
            "\nNumber of classes", label_encoder.classes_.shape[0]
            )
    return (persistence_diagrams_to_sequence(tensor_dict),
            torch.tensor(x_features, dtype=torch.float),
            torch.tensor(y))

def load_data_as_tensor(
        dataset_name: str = "PROTEINS",
        path_dataset: str = "graph_data",
        verbose: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_pds_dict, x_features, y = load_data(dataset_name)

    # Compute the max number of points in the persistence diagrams
    max_number_of_points = max([x_pd.shape[0]
                                for _, x_pd in x_pds_dict.items()])  # type: ignore

    x_pds = pad_pds(x_pds_dict, max_number_of_points)

    return x_pds, x_features, y

def load_augmented_data_as_tensor(
        dataset_name: str = "PROTEINS",
        path_dataset: str = "graph_data",
        verbose: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dataset_ != "PROTEINS":
        raise NotImplementedError()
    x_pds, x_features, y

    # Indices of graphs that are used for the graph generation per class
    idx_class = {}

    for class_ in ['1', '2']:
        with open(os.path.join('graph_data', dataset_name + '_' + class_, 'gid.txt')) as f:
            lines = f.readlines()
            idx_class[class_] = [int(idx) for idx in lines[0].split(', ')]

        # load the augmented training data for class_
        x_pds_dict_aug, x_features_aug, y_aug = load_data(dataset_name + "_" + class_)
        # load_data only encodes a single class
        if class_ == '2':
            y_aug += 1
        x_pds_aug = pad_pds(x_pds_dict_aug, max_number_of_points)

        x_pds = torch.cat([x_pds, x_pds_aug], axis=0)
        # here might be a problem
        x_features = torch.cat([x_features[:, :x_features_aug.shape[1]], x_features_aug], axis=0)
        y = torch.cat([y, y_aug], axis=0)

def pad_pds(x_pds_dict, max_number_of_points):
    """Pad persistence diagrams to make them the same size

    Args:
        x_pds_dict (dict): dictionary of persistence diagrams. The
            keys must be from 0 to size of x_pds_dict.
        max_number_of_points (int): padding size. If one of the persistence
            diagrams in x_pds_dict has more than max_number_of_points points,
            a runtime error will be thrown.

    Returns:
        torch.Tensor: padded persistence diagrams
    """
    # transform x_pds to a single tensor with tailing zeros
    num_types = x_pds_dict[0].shape[1] - 2
    num_graphs = len(x_pds_dict.keys())  # type: ignore

    x_pds = torch.zeros((num_graphs, max_number_of_points, num_types + 2))

    for idx, x_pd in x_pds_dict.items():  # type: ignore
        if x_pd.shape[0] > max_number_of_points:
            raise RuntimeError("max_number_of_points is smaller than points in x_pd")
        x_pds[idx, :x_pd.shape[0], :] = x_pd

    return x_pds

def diagram_to_tensor(
    tensor_dict_per_type: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
    """Convert dictionary of diagrams for fixed type to tensor representation
    with tailing zeros

    Args:
        tensor_dict (Dict[str, torch.Tensor]): Dictionary of persistence
            diagrams of a fixed type. Keys are strings of graph indices and
            values are tensor representations of persistence diagrams.
            The keys are assumed to be in range(len(tensor_dict_per_type)).

    Returns:
        torch.Tensor: [description]
    """
    try:
        assert all([int(k) in range(len(tensor_dict_per_type))
                    for k in tensor_dict_per_type.keys()])
    except AssertionError:
        print("Tensor dictionary should contain all keys in",
              "range(len(tensor_dict_per_type))")
        raise
    max_number_of_points = max([v.shape[0]
                                for v in tensor_dict_per_type.values()])

    diagram_tensor = torch.zeros((
                            len(tensor_dict_per_type),
                            max_number_of_points,
                            2
                        ))
    for graph_idx, diagram in tensor_dict_per_type.items():
        # number of points in persistence diagram
        npts = tensor_dict_per_type[graph_idx].shape[0]
        diagram_tensor[int(graph_idx)][:npts] = tensor_dict_per_type[graph_idx]

    return diagram_tensor
