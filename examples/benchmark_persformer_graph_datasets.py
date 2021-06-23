# %% [markdown]
# ## Benchmarking PersFormer on the graph datasets.
# We will compare the accuracy on the graph datasets of our SetTransformer
# based on PersFormer with the perslayer introduced in the paper:
# https://arxiv.org/abs/1904.09378

# %% [markdown]
# ## Benchmarking MUTAG
# We will compare the test accuracies of PersLay and PersFormer on the MUTAG
# dataset. It consists of 188 graphs categorised into two classes.
# We will train the PersFormer on the same input features as PersFormer to
# get a fair comparison.
# The features PersLay is trained on are the extended persistence diagrams of
# the vertices of the graph filtered by the heat kernel signature (HKS)
# at time t=10.
# The maximum (wrt to the architecture and the hyperparameters) mean test
# accuracy of PersLay is 89.8(±0.9) and the train accuracy with the same
# model and the same hyperparameters is 92.3.
# They performed 10-fold evaluation, i.e. splitting the dataset into
# 10 equally-sized folds and then record the test accuracy of the i-th
# fold and training the model on the 9 other folds.

# %%
# Import libraries:
from typing import Tuple, Dict, List
import numpy as np  # typing: ignore
import random
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from os.path import join, isfile
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore
import h5py  # type: ignore
from gdeep.topology_layers import SetTransformer
# for loading extended persistence diagrams that are in hdf5 format
from sam import SAM

# %%
DATASET_NAME = "NCI1"
pers_only = True
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# %%


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


# %%
# Load extended persistence diagrams and additional features


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


x_pds_dict, x_features, y = load_data(DATASET_NAME)
if pers_only:
    x_features = 0.0 * x_features
# %%
# Test persistence_diagrams_to_sequence with MUTAG dataset


def random_compatibility_test(n_trials: int = 10) -> None:
    """Randomly check if persistence diagram is correctly converted to
    sequence.

    Raises:
        Exception: Assertion error.

    Returns:
        None
    """
    filename = join("graph_data",
                    "MUTAG",
                    "MUTAG" + ".hdf5")
    diagrams_file = h5py.File(filename, "r")

    # load_data with `load_data` methode
    seq_pd, _, _ = load_data("MUTAG")

    for _ in range(n_trials):

        type_ = random.choice(list(diagrams_file.keys()))

        type_index = list(diagrams_file.keys()).index(type_)

        graph_number = random.choice(list(diagrams_file[type_].keys()))

        # find indices that belong to type_
        idx = seq_pd[int(graph_number)][:, 2 + type_index] == 1.0

        computed_pts = seq_pd[int(graph_number)][idx][:, :2]

        original_pts = torch.tensor(diagrams_file[type_][graph_number])

        try:
            assert torch.allclose(original_pts, computed_pts)
        except AssertionError:
            raise AssertionError("persistence_diagrams_to_sequence does not" +
                                 "return the right sequence tensor")


random_compatibility_test()
# %%
# Test persistence_diagrams_to_sequence


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
         [0.2015, 0.3533, 0.0000, 1.0000]])}

for i in range(3):
    try:
        assert(torch.allclose(output[i], expected_output[i]))
    except AssertionError:
        print("expected:\n", expected_output[i])
        print("actual:\n", output[i])
        raise AssertionError("persistence_diagrams_to_sequence does not match")


# %%

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


# check if tensorised diagrams have the correct shape

def test_diagram_to_tensor():
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


test_diagram_to_tensor()

# %%
# Set up dataset and dataloader

# transform x_pds to a single tensor with tailing zeros
num_types = x_pds_dict[0].shape[1] - 2
num_graphs = len(x_pds_dict.keys())  # type: ignore

max_number_of_points = max([x_pd.shape[0]
                            for _, x_pd in x_pds_dict.items()])  # type: ignore

x_pds = torch.zeros((num_graphs, max_number_of_points, num_types + 2))

for idx, x_pd in x_pds_dict.items():  # type: ignore
    x_pds[idx, :x_pd.shape[0], :] = x_pd

# https://discuss.pytorch.org/t/make-a-tensordataset-and-dataloader
# -with-multiple-inputs-parameters/26605
total_size = x_pds.shape[0]

graph_ds = TensorDataset(x_pds, x_features, y)

train_size = int(total_size * 0.8)
graph_ds_train, graph_ds_val = torch.utils.data.random_split(
                                                    graph_ds,
                                                    [train_size,
                                                     total_size - train_size])

graph_dl_train = DataLoader(
    graph_ds_train,
    num_workers=4,
    batch_size=64,
    shuffle=True
)

graph_dl_val = DataLoader(
    graph_ds_val,
    num_workers=4,
    batch_size=64,
    shuffle=False
)


# %%
# Train Persformer on MUTAG dataset

class GraphClassifier(nn.Module):
    """Classifier for Graphs using persistence features and additional
    features. The vectorization is based on a set transformer.
    """
    def __init__(self,
                 num_features,
                 dim_input=6,
                 num_outputs=1,
                 dim_output=10,
                 num_classes=2):
        super(GraphClassifier, self).__init__()
        self.st = SetTransformer(
            dim_input=dim_input,
            num_outputs=num_outputs,
            dim_output=dim_output
            )
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(dim_output + num_features)
        self.ff_1 = nn.Linear(dim_output + num_features, 50)
        self.ff_2 = nn.Linear(50, 20)
        self.ff_3 = nn.Linear(20, num_classes)

    def forward(self, x_pd: Tensor, x_feature: Tensor) -> Tensor:
        """Forward pass of the graph classifier.
        The persistence features are encoded with a set transformer
        and concatenated with the feature vector. These concatenated
        features are used for classification using a fully connected
        feed -forward layer.

        Args:
            x_pd (Tensor): persistence diagrams of the graph
            x_feature (Tensor): additional graph features
        """
        pd_vector = self.st(x_pd)
        features_stacked = torch.hstack((pd_vector, x_feature))
        x = self.bn(features_stacked)
        x = nn.ReLU()(self.ff_1(x))
        x = nn.ReLU()(self.ff_2(x))
        x = self.ff_3(x)
        return x


gc = GraphClassifier(
        num_features=graph_ds_train[0][1].shape[0],
        dim_input=graph_ds_train[0][0].shape[1],
        num_outputs=1,
        dim_output=50)

# %%


def compute_accuracy(
                    model: nn.Module,
                    dl,
                    use_cuda: bool = False
                  ) -> Tuple[int, float, float]:
    """Print the accuracy of the network on the dataset
    provided by the data loader.

    Args:
        model (nn.Module): Model to be evaluated.
        dataloader ([type]): dataloader of the dataset the model is being
            evaluated.
        use_cuda (bool, optional): If the model is on GPU. Defaults to False.
    """
    model.eval()
    correct = 0
    total = 0

    for x_pd, x_feature, label in dl:
        if use_cuda:
            print("cuda")
            x_pd, x_feature, label = x_pd.cuda(), x_feature.cuda(),\
                label.cuda()
        outputs = model(x_pd, x_feature)
        loss = nn.CrossEntropyLoss()(outputs, label)
        _, predictions = torch.max(outputs.squeeze(1), 1)
        total += label.size(0)
        correct += (predictions == label).sum().item()

    return (total, 100 * correct/total, loss.item())


def train(model, train_dl, val_dl, criterion=nn.CrossEntropyLoss(),
          lr: float = 1e-3, num_epochs=10,
          verbose=False):
    if use_cuda:
        model = nn.DataParallel(model)
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []
    for epoch in range(num_epochs):
        model.train()
        loss_per_epoch = 0
        for batch_idx, (x_pd, x_feature, label) in enumerate(train_dl):
            # transfer to GPU
            if use_cuda:
                x_pd, x_feature, label = x_pd.cuda(), x_feature.cuda(),\
                    label.cuda()
            loss = criterion(model(x_pd, x_feature), label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_per_epoch += loss.item()
        losses.append(loss_per_epoch)
        if verbose:
            # print train loss, test and model accuracy
            print("epoch:", epoch, "loss:", loss_per_epoch)
            train_total, train_accuracy, _ = compute_accuracy(model,
                                                              train_dl
                                                              )
            print('Test',
                  'accuracy of the network on the', train_total,
                  'diagrams: %8.2f %%' % train_accuracy
                  )
            train_accuracies.append(train_accuracy)
            if val_dl is not None:
                val_total, val_accuracy, val_loss = compute_accuracy(model,
                                                                     val_dl)
                print('Val',
                      'accuracy of the network on the', val_total,
                      'diagrams: %8.2f %%' % val_accuracy
                      )
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
    return losses, val_losses, train_accuracies, val_accuracies
# %%


losses, val_losses, train_accuracies, val_accuracies = train(gc,
                                                             graph_dl_train,
                                                             graph_dl_val,
                                                             verbose=True,
                                                             num_epochs=100)


# %%
# plot losses
plt.plot(losses, label='train_loss')
plt.plot(val_losses, label='val_loss')
plt.legend()
plt.show()

# %%
# plot accuracies
plt.plot(train_accuracies, label='train_acc')
plt.plot(val_accuracies, label='val_acc')
plt.plot([72.3]*len(train_accuracies), label='PersLay')
plt.legend()
plt.show()
# %%
# Use SAM optimizer

gc_sam = GraphClassifier(
        num_features=graph_ds_train[0][1].shape[0],
        dim_input=graph_ds_train[0][0].shape[1],
        num_outputs=1,
        dim_output=50)


def sam_train(model, train_dl, val_dl, criterion=nn.CrossEntropyLoss(),
              lr: float = 1e-3, num_epochs=10,
              verbose=False):
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, momentum=0.9)
    gc.train()
    losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []
    for epoch in range(num_epochs):
        loss_per_epoch = 0
        for batch_idx, (x_pd, x_feature, label) in enumerate(train_dl):

            # first forward-backward pass
            loss = criterion(model(x_pd, x_feature), label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)
            # second forward-backward pass
            criterion(model(x_pd, x_feature), label.long()).backward()
            optimizer.second_step(zero_grad=True)

            loss_per_epoch += loss.item()
        losses.append(loss_per_epoch)
        if verbose:
            # print train loss, test and model accuracy
            print("epoch:", epoch, "loss:", loss_per_epoch)
            train_total, train_accuracy, _ = compute_accuracy(model,
                                                              train_dl
                                                              )
            print('Test',
                  'accuracy of the network on the', train_total,
                  'diagrams: %8.2f %%' % train_accuracy
                  )
            train_accuracies.append(train_accuracy)
            if val_dl is not None:
                val_total, val_accuracy, val_loss = compute_accuracy(model,
                                                                     val_dl)
                print('Val',
                      'accuracy of the network on the', val_total,
                      'diagrams: %8.2f %%' % val_accuracy
                      )
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
    return losses, val_losses, train_accuracies, val_accuracies

# %%


(losses,
 val_losses,
 train_accuracies,
 val_accuracies) = sam_train(gc_sam,
                             graph_dl_train,
                             graph_dl_val,
                             verbose=True,
                             num_epochs=100)
 
 