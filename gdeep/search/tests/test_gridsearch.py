import torch
from torch import nn
import torchvision.models as models
from gdeep.pipeline import Pipeline
from torch.utils.data.sampler import SubsetRandomSampler
from gdeep.search.gridsearch import Gridsearch
from torch.optim import SGD, Adam, RMSprop
from gdeep.pipeline import Pipeline
from torch.utils.tensorboard import SummaryWriter


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


writer = SummaryWriter()
loss_fn = nn.CrossEntropyLoss()
model = nn.Sequential(models.resnet18(pretrained=True), nn.Linear(1000,10))
dl = TorchDataLoader(name="CIFAR10")
train_indices = list(range(32*10))

dl_tr, dl_temp = dl.build_dataloader(batch_size=32,
                                     sampler=SubsetRandomSampler(train_indices))

test_indices = [32*10 + x for x in list(range(32*2))]

dl_ts, dl_temp = dl.build_dataloader(batch_size=32, sampler=SubsetRandomSampler(test_indices))

dl_val = dl_ts

def test_gridsearch():
    pipe = Pipeline(model, [dl_tr, dl_val, dl_ts], loss_fn, writer)

    search = Gridsearch(pipe, "loss", 2)
    opimizers_params = {"n_epochs": 1}
    search.start([SGD, Adam], opimizers_params)
