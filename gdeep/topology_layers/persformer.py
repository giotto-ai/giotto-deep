# from https://github.com/juho-lee/set_transformer/blob/master/max_regression_demo.ipynb  # noqa: E501

import torch
import torch.nn as nn
from torch import Tensor
from gdeep.topology_layers.modules import ISAB, PMA, SAB  # type: ignore


class SmallDeepSet(nn.Module):
    def __init__(self, pool="max"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )
        self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        x = self.dec(x)
        return x


class SetTransformer(nn.Module):
    """ Vanilla SetTransformer from
    https://github.com/juho-lee/set_transformer/blob/master/main_pointcloud.py
    """
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=40,  # number of classes
        num_inds=32,  # number of induced points, see  Set Transformer paper
        dim_hidden=128,
        num_heads=4,
        ln=True,  # use layer norm
    ):
        """Init of SetTransformer.

        Args:
            dim_input (int, optional): Dimension of input data for each
            element in the set. Defaults to 3.
            num_outputs (int, optional): Number of outputs. Defaults to 1.
            dim_output (int, optional): Number of classes. Defaults to 40.
            dim_hidden (int, optional): Number of induced points, see  Set
                Transformer paper. Defaults to 128.
            num_heads (int, optional): Number of attention heads. Defaults
                to 4.
            ln (bool, optional): If `True` layer norm will not be applied.
                Defaults to False.
        """
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Batch tensor of shape
                [batch, sequence_length, dim_in]

        Returns:
            torch.Tensor: Tensor with predictions of the `dim_output` classes.
        """
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze()

    def num_params(self) -> int:
        """Returns number of trainable parameters.

        Returns:
            int: Number of trainable parameters.
        """
        total_params = 0
        for parameter in self.parameters():
            total_params += parameter.nelement()
        return total_params


class SelfAttentionSetTransformer(nn.Module):
    """ Vanilla SetTransformer from
    https://github.com/juho-lee/set_transformer/blob/master/main_pointcloud.py
    """
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=40,  # number of classes
        dim_hidden=128,
        num_heads=4,
        ln=False,  # use layer norm
    ):
        """Init of SetTransformer.

        Args:
            dim_input (int, optional): Dimension of input data for each
            element in the set. Defaults to 3.
            num_outputs (int, optional): Number of outputs. Defaults to 1.
            dim_output (int, optional): Number of classes. Defaults to 40.
            dim_hidden (int, optional): Number of induced points, see  Set
                Transformer paper. Defaults to 128.
            num_heads (int, optional): Number of attention heads. Defaults
                to 4.
            ln (bool, optional): If `True` layer norm will not be applied.
                Defaults to False.
        """
        super(SelfAttentionSetTransformer, self).__init__()
        self.enc = nn.Sequential(
            SAB(dim_input, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Batch tensor of shape
                [batch, sequence_length, dim_in]

        Returns:
            torch.Tensor: Tensor with predictions of the `dim_output` classes.
        """
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze()

    def num_params(self) -> int:
        """Returns number of trainable parameters.

        Returns:
            int: Number of trainable parameters.
        """
        total_params = 0
        for parameter in self.parameters():
            total_params += parameter.nelement()
        return total_params

class GraphClassifier(nn.Module):
    """Classifier for Graphs using persistence features and additional
    features. The vectorization is based on a set transformer.
    """
    def __init__(self,
                 num_features,
                 dim_input=6,
                 num_outputs=1,
                 dim_output=50,
                 num_classes=2,
                 ln=True,
                 num_heads=4,
                 use_induced_attention=False
                ):
        super().__init__()
        if use_induced_attention:
            self.st = SetTransformer(
                dim_input=dim_input,
                num_outputs=num_outputs,
                dim_output=dim_output,
                ln=ln,
                num_heads=num_heads
                )
        else:
            self.st = SelfAttentionSetTransformer(
                dim_input=dim_input,
                num_outputs=num_outputs,
                dim_output=dim_output,
                ln=ln
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
        #print(pd_vector.shape, x_feature.shape)
        features_stacked = torch.hstack((pd_vector, x_feature))
        x = self.ln(features_stacked)
        x = nn.ReLU()(self.ff_1(x))
        x = nn.ReLU()(self.ff_2(x))
        x = self.ff_3(x)
        return x

# def train(
#         model,
#         num_epochs: int = 10,
#         lr: float = 1e-3,
#         train_dataloader: Optional[DataLoader] = None,
#         validation_dataloader: DataLoader = None,
#         use_cuda: bool = False,
#         verbose: bool = False,
#         ) -> List[float]:
#     """Custom training loop for Set Transformer on the dataset ``

#     Args:
#         model (nn.Module): Set Transformer model to be trained
#         num_epochs (int, optional): Number of training epochs.
#           Defaults to 10.
#         lr (float, optional): Learning rate for training. Defaults to 1e-3.
#         verbose (bool, optional): Print training loss, training accuracy and
#             validation if set to True. Defaults to False.

#     Returns:
#         List[float]: List of training losses
#     """
#     try:
#         assert train_dataloader is not None
#     except AssertionError:
#         print("train_dataloader is not set")
#     if use_cuda:
#         model = nn.DataParallel(model)
#         model = model.cuda()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
#     losses: List[float] = []
#     # training loop
#     for epoch in range(num_epochs):
#         model.train()
#         loss_per_epoch = 0
#         for x_batch, y_batch in train_dataloader:  # type: ignore
#             # transfer to GPU
#             if use_cuda:
#                 x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
#             loss = criterion(model(x_batch), y_batch.long())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             loss_per_epoch += loss.item()
#         losses.append(loss_per_epoch)
#         if verbose:
#             # print train loss, test and model accuracy
#             print("epoch:", epoch, "loss:", loss_per_epoch)
#             test_total, test_accuracy = compute_accuracy(model,
#                                                          train_dataloader)
#             print('Test',
#                   'accuracy of the network on the', test_total,
#                   'diagrams: %8.2f %%' % test_accuracy
#                   )
#             if validation_dataloader is not None:
#                 test_total, test_accuracy = compute_accuracy(model,
#                                                              train_dataloader)
#                 print('Test',
#                       'accuracy of the network on the', test_total,
#                       'diagrams: %8.2f %%' % test_accuracy
#                       )
#     return losses


# def compute_accuracy(
#                     model: nn.Module,
#                     dataloader,
#                     use_cuda: bool = False
#                   ) -> Tuple[int, float]:
#     """Print the accuracy of the network on the dataset
#     provided by the data loader.

#     Args:
#         model (nn.Module): Model to be evaluated.
#         dataloader ([type]): dataloader of the dataset the model is being
#             evaluated.
#         use_cuda (bool, optional): If the model is on GPU. Defaults to False.
#     """
#     model.eval()
#     correct = 0
#     total = 0

#     for x_batch, y_batch in dataloader:
#         # transform the data to GPU
#         if use_cuda:
#             x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
#         outputs = model(x_batch).squeeze(1)
#         _, predictions = torch.max(outputs, 1)
#         total += y_batch.size(0)
#         correct += (predictions == y_batch).sum().item()
#     return (total, 100 * correct/total)
