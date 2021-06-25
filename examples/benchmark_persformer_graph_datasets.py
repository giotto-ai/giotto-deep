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
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt  # type: ignore

from gdeep.topology_layers import load_data, SetTransformer, train

# for SAM training
from gdeep.topology_layers import sam_train

# %%
DATASET_NAME = "MUTAG"
pers_only = True  
# only use the persistence diagrams as features not the spectral features
train_size = 0.8  # ratio train size to total size of dataset

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# %%
# Load extended persistence diagrams and additional features


x_pds_dict, x_features, y = load_data(DATASET_NAME)

# if `pers_only` set spectral features to 0
if pers_only:
    x_features = 0.0 * x_features

# %%
# Padding persistence diagrams to make them the same size

# transform x_pds to a single tensor with tailing zeros
num_types = x_pds_dict[0].shape[1] - 2
num_graphs = len(x_pds_dict.keys())  # type: ignore

max_number_of_points = max([x_pd.shape[0]
                            for _, x_pd in x_pds_dict.items()])  # type: ignore

x_pds = torch.zeros((num_graphs, max_number_of_points, num_types + 2))

for idx, x_pd in x_pds_dict.items():  # type: ignore
    x_pds[idx, :x_pd.shape[0], :] = x_pd

# %%
# Set up dataset and dataloader
# create the datasets
# https://discuss.pytorch.org/t/make-a-tensordataset-and-dataloader
# -with-multiple-inputs-parameters/26605
total_size = x_pds.shape[0]

graph_ds = TensorDataset(x_pds, x_features, y)

train_size = int(total_size * train_size)
graph_ds_train, graph_ds_val = torch.utils.data.random_split(
                                                    graph_ds,
                                                    [train_size,
                                                     total_size - train_size])

# create data loaders
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
print('val_balance', next(iter(graph_dl_val))[-1].sum() / 38.0)

train_balance = 0
for _, _, y_batch in graph_dl_train:
    train_balance += y_batch.sum()
print('train_balance', train_balance / 150)
# %%
# Define Model architecture for the graph classifier


class GraphClassifier(nn.Module):
    """Classifier for Graphs using persistence features and additional
    features. The vectorization is based on a set transformer.
    """
    def __init__(self,
                 num_features,
                 dim_input=6,
                 num_outputs=1,
                 dim_output=50,
                 num_classes=2):
        super(GraphClassifier, self).__init__()
        self.st = SetTransformer(
            dim_input=dim_input,
            num_outputs=num_outputs,
            dim_output=dim_output,
            ln=True
            )
        self.num_classes = num_classes
        self.ln = nn.LayerNorm(dim_output + num_features)
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
        x = self.ln(features_stacked)
        x = nn.ReLU()(self.ff_1(x))
        x = nn.ReLU()(self.ff_2(x))
        x = self.ff_3(x)
        return x


# define graph classifier
gc = GraphClassifier(
        num_features=graph_ds_train[0][1].shape[0],
        dim_input=graph_ds_train[0][0].shape[1],
        num_outputs=1,
        dim_output=50)

# %%

# train the model and return losses and accuracies information
losses, val_losses, train_accuracies, val_accuracies = train(gc,
                                                             graph_dl_train,
                                                             graph_dl_val,
                                                             lr=1e-4,
                                                             verbose=True,
                                                             num_epochs=50,
                                                             use_cuda=use_cuda)


# %%
# plot losses
plt.plot(losses, label='train_loss')
plt.plot([4.0 * x for x in val_losses], label='4 * val_loss')
plt.legend()
plt.show()

# %%
# plot accuracies
plt.plot(train_accuracies, label='train_acc')
plt.plot(val_accuracies, label='val_acc')
plt.plot([85.1]*len(train_accuracies), label='PersLay PD only')
plt.legend()
plt.show()
# %%
# Use SAM optimizer

# TODO: Implement properly with batch norm layer
# there might be a problem
gc_sam = GraphClassifier(
        num_features=graph_ds_train[0][1].shape[0],
        dim_input=graph_ds_train[0][0].shape[1],
        num_outputs=1,
        dim_output=50)


# %%


(losses,
 val_losses,
 train_accuracies,
 val_accuracies) = sam_train(gc,
                             graph_dl_train,
                             graph_dl_val,
                             verbose=True,
                             num_epochs=100)
# %%
