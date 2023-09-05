from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
import numpy as np

from gdeep.trainer import Trainer
from gdeep.trainer import RegularizedTrainer
from gdeep.trainer import TihonovRegularizer
from gdeep.models import FFNet
from gdeep.search import GiottoSummaryWriter
from gdeep.data.datasets import FromArray, DataLoaderBuilder
from gdeep.search import clean_up_files

from gdeep.utility.custom_types import Tensor


class MyDataset(Dataset):
    def __init__(self):
        self.x = []
        for _ in range(100):
            self.x.append((torch.rand(1, np.random.randint(2, 4))).to(torch.float))
        self.y = np.array(
            np.random.randint(2, size=100 * 2).reshape(-1, 2), dtype=np.int64
        )

    def __len__(self):
        return 100

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        return self.x[item], self.y[item]


def collate_fn(batch_tuple: List):
    # print(batch_tuple)
    target = torch.zeros(len(batch_tuple), 4)
    label = torch.zeros(len(batch_tuple), 2).to(torch.long)
    for i, batch in enumerate(batch_tuple):
        source = batch[0]
        target[i, : len(source[-1])] = source
        label[i] = torch.tensor(batch[1]).to(torch.long)
    return target, label


# model
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.seqmodel = nn.Sequential(nn.Flatten(), FFNet(arch=(4, 5, 4)))

    def forward(self, x):
        return self.seqmodel(x).reshape(-1, 2, 2)


def test_regularization_negative():
    """
    This simple test verifies normal Trainer does not attempt regularization
    """
    model = Model1()
    # dataloaders
    X = np.array(np.random.rand(100, 4), dtype=np.float32)  # type: ignore
    y = np.array(np.random.randint(2, size=100 * 2).reshape(-1, 2), dtype=np.int64)
    dl_tr, *_ = DataLoaderBuilder([FromArray(X, y)]).build([{"batch_size": 23}])

    # loss function
    loss_fn = nn.CrossEntropyLoss()
    # tb writer
    writer = GiottoSummaryWriter()
    # pipeline
    pipe = Trainer(model, [dl_tr, None], loss_fn, writer)  # type: ignore
    assert pipe.regularize == False


def test_regularization_positive():
    """
    This simple test verifies RegularizedTrainer does attempt regularization
    """
    model = Model1()
    # dataloaders
    X = np.array(np.random.rand(100, 4), dtype=np.float32)  # type: ignore
    y = np.array(np.random.randint(2, size=100 * 2).reshape(-1, 2), dtype=np.int64)
    dl_tr, *_ = DataLoaderBuilder([FromArray(X, y)]).build([{"batch_size": 23}])

    # loss function
    loss_fn = nn.CrossEntropyLoss()
    # tb writer
    writer = GiottoSummaryWriter()
    reg = TihonovRegularizer()

    # pipeline
    pipe = RegularizedTrainer(regularizer=reg, reg_params={"p": 2}, model=model, dataloaders=[dl_tr, None], loss_fn=loss_fn, writer=writer)  # type: ignore
    assert pipe.regularize == True


def test_RegTrainer_from_array():
    """
    Test to Verify that the regularization pipeline works
    """
    model = Model1()
    # dataloaders
    X = np.array(np.random.rand(100, 4), dtype=np.float32)  # type: ignore
    y = np.array(np.random.randint(2, size=100 * 2).reshape(-1, 2), dtype=np.int64)
    dl_tr, *_ = DataLoaderBuilder([FromArray(X, y)]).build([{"batch_size": 23}])

    # loss function
    loss_fn = nn.CrossEntropyLoss()
    # tb writer
    writer = GiottoSummaryWriter()
    # pipeline
    reg = TihonovRegularizer()
    # pipeline
    pipe = RegularizedTrainer(regularizer=reg, reg_params={"p": 2}, model=model, dataloaders=[dl_tr, None], loss_fn=loss_fn, writer=writer)  # type: ignore
    # then one needs to train the model using the pipeline!
    pipe.lamda = 1
    pipe.train(SGD, 2, True, {"lr": 0.001}, n_accumulated_grads=2)
