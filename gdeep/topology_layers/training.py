from typing import Tuple, List, Callable, Union, Any
import torch
import torch.nn as nn
from ..utility.optimisation import SAM


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
          verbose=False,
          use_cuda: bool = False,
          use_regularization=False,
          optimizer: Union[Callable[[torch.Tensor], torch.optim.Optimizer],
                           Any] = None):
    if use_cuda:
        model = nn.DataParallel(model)
        model = model.cuda()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optimizer(model.parameters())

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
            if use_regularization:
                l2_lambda = 0.01
                l2_reg = torch.tensor(0.)
                if use_cuda:
                    l2_reg = l2_reg.cuda()
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_per_epoch += loss.item()
        if verbose:
            # print train loss, test and model accuracy
            train_total, train_accuracy, train_loss = compute_accuracy(model,
                                                                       train_dl,
                                                                       use_cuda)
            losses.append(train_loss)
            print("epoch:", epoch, "loss:", train_loss)
            print('Train', train_total, train_accuracy)
            train_accuracies.append(train_accuracy)
            if val_dl is not None:
                val_total, val_accuracy, val_loss = compute_accuracy(model,
                                                                     val_dl,
                                                                     use_cuda)
                print('Val', val_total, val_accuracy)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
    return losses, val_losses, train_accuracies, val_accuracies


def sam_train(model, train_dl, val_dl, criterion=nn.CrossEntropyLoss(),
              lr: float = 1e-3, num_epochs=10,
              verbose=False, use_cuda=False, use_regularization=False,
              optimizer: Union[Callable[[torch.Tensor], torch.optim.Optimizer],
                           Any] = None):
    if use_regularization:
        raise NotImplementedError("L2-regularization is not implemented" +
                                  "for SAM training.")
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, momentum=0.9)
    model.train()
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
                                                              train_dl,
                                                              use_cuda
                                                              )
            print('Train', train_total, train_accuracy)
            train_accuracies.append(train_accuracy)
            if val_dl is not None:
                val_total, val_accuracy, val_loss = compute_accuracy(model,
                                                                     val_dl,
                                                                     use_cuda)
                print('Val', val_total, val_accuracy)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
    return losses, val_losses, train_accuracies, val_accuracies


def print_accuracy(type_: str, total: int, accuracy: float) -> None:
    print(type_,
          'accuracy of the network on the', total,
          'diagrams: %8.2f %%' % accuracy
          )


def compute_accuracy_vec(
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

    for x_pd, label in dl:
        if use_cuda:
            x_pd, label = x_pd.cuda(),\
                label.cuda()
        outputs = model(x_pd)
        loss = nn.CrossEntropyLoss()(outputs, label)
        _, predictions = torch.max(outputs.squeeze(1), 1)
        total += label.size(0)
        correct += (predictions == label).sum().item()

    return (total, 100 * correct/total, loss.item())


def train_vec(model, train_dl, val_dl, criterion=nn.CrossEntropyLoss(),
          lr: float = 1e-3, num_epochs=10,
          verbose=False,
          use_cuda: bool = False,
          use_regularization=False,
          optimizer: Union[Callable[[torch.Tensor], torch.optim.Optimizer],
                           Any] = None):
    if use_cuda:
        model = nn.DataParallel(model)
        model = model.cuda()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optimizer(model.parameters())

    losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []
    for epoch in range(num_epochs):
        model.train()
        loss_per_epoch = 0
        for batch_idx, (x_pd, label) in enumerate(train_dl):
            # transfer to GPU
            if use_cuda:
                x_pd, label = x_pd.cuda(),\
                    label.cuda()
            
            loss = criterion(model(x_pd), label.long())
            if use_regularization:
                l2_lambda = 0.01
                l2_reg = torch.tensor(0.)
                if use_cuda:
                    l2_reg = l2_reg.cuda()
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_per_epoch += loss.item()
        if verbose:
            # print train loss, test and model accuracy
            train_total, train_accuracy, train_loss = compute_accuracy_vec(model,
                                                                       train_dl,
                                                                       use_cuda)
            losses.append(train_loss)
            print("epoch:", epoch, "loss:", train_loss)
            print('Train', train_total, train_accuracy)
            train_accuracies.append(train_accuracy)
            if val_dl is not None:
                val_total, val_accuracy, val_loss = compute_accuracy_vec(model,
                                                                     val_dl,
                                                                     use_cuda)
                print('Val', val_total, val_accuracy)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
    return losses, val_losses, train_accuracies, val_accuracies