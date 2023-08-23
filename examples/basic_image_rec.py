import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
from gtda.diagrams import BettiCurve
from gtda.plotting import plot_betti_surfaces

from gdeep.data.datasets import DatasetBuilder, DataLoaderBuilder
from gdeep.models import FFNet
from gdeep.visualization import persistence_diagrams_of_activations
from gdeep.data.preprocessors import ToTensorImage
from gdeep.trainer import Trainer
from gdeep.models import ModelExtractor
from gdeep.analysis.interpretability import Interpreter
from gdeep.visualization import Visualiser
from gdeep.search import GiottoSummaryWriter

writer = GiottoSummaryWriter()

db = DatasetBuilder(name="CIFAR10")
ds_tr, ds_val, ds_ts = db.build(download=True, root="cifar")
NUMBER_OF_CLASSES = 10

# Preprocessing


transformation = ToTensorImage((32, 32))
transformation.fit_to_dataset(
    ds_tr
)  # this is useless for this transformation, but in general this is the API

transformed_ds_tr = transformation.attach_transform_to_dataset(ds_tr)
transformed_ds_val = transformation.attach_transform_to_dataset(ds_val)
transformed_ds_ts = transformation.attach_transform_to_dataset(ds_ts)

# use only 320 images from cifar10 for training
train_indices = list(range(32 * 10))
val_indices = list(range(32 * 5))
test_indices = list(range(32 * 5))
dl_tr, dl_val, dl_ts = DataLoaderBuilder(
    (transformed_ds_tr, transformed_ds_val, transformed_ds_ts)
).build(
    (
        {"batch_size": 32, "sampler": SubsetRandomSampler(train_indices)},
        {"batch_size": 32, "sampler": SubsetRandomSampler(val_indices)},
        {"batch_size": 32, "sampler": SubsetRandomSampler(test_indices)},
    )
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

print(model)

# define the loss function
loss_fn = nn.CrossEntropyLoss()

# initilise the trainer class
pipe = Trainer(model, (dl_tr, dl_ts), loss_fn, writer)

# train the model
pipe.train(
    SGD,
    3,
    False,
    {"lr": 0.01},
    {"batch_size": 32, "sampler": SubsetRandomSampler(train_indices)},
    profiling=True
)

pipe.evaluate_classification(NUMBER_OF_CLASSES)