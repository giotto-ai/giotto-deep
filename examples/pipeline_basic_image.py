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
from gdeep.trainer.trainer import Trainer, Parallelism, ParallelismType
from gdeep.models import ModelExtractor
from gdeep.analysis.interpretability import Interpreter
from gdeep.visualization import Visualiser
from gdeep.search import GiottoSummaryWriter
import argparse

parser = argparse.ArgumentParser(description='Pipeline enabling')
parser.add_argument('--pipeline', default=False, action='store_true')
args = parser.parse_args()
pipeline_enabling = args.pipeline

if pipeline_enabling:
    print("Pipeline as been enabled")
else:
    print("Pipeline is not enabled")

writer = GiottoSummaryWriter()

db = DatasetBuilder(name="CIFAR10")
ds_tr, ds_val, ds_ts = db.build(download=True)
NUMBER_OF_CLASSES = 10

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

# define the loss function
loss_fn = nn.CrossEntropyLoss()

# initilise the trainer class
pipe = Trainer(model, (dl_tr, dl_ts), loss_fn, writer)

devices = list(range(torch.cuda.device_count()))
parallel = Parallelism(ParallelismType.PIPELINE,
                           devices,
                           len(devices),
                           pipeline_chunks=2,
                           config_mha=[{'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True},
            {'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True},
            {'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True},
            {'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True},
            {'embed_dim': 16, 'num_heads': 8, 'dropout': 0.1, 'batch_first': True}])

if (pipeline_enabling):
    # train the model
    pipe.train(
        SGD,
        3,
        False,
        {"lr": 0.01},
        {"batch_size": 32, "sampler": SubsetRandomSampler(train_indices)},
        parallel=parallel
    )

else:
    pipe.train(
        SGD,
        3,
        False,
        {"lr": 0.01},
        {"batch_size": 32, "sampler": SubsetRandomSampler(train_indices)},
    )


pipe.evaluate_classification(NUMBER_OF_CLASSES)

# initialise the interpreter
inter = Interpreter(pipe.model, method="GuidedGradCam")

# define a signle datum
datum = next(iter(dl_tr))[0][0].reshape(1, 3, 32, 32)

# define the layer of which we are interested in displaying the features
layer = pipe.model.conv2

# we will test against this class
class_ = 0

# interpret the image
output = inter.interpret(datum, class_, layer)

# visualise the interpreter
vs = Visualiser(pipe)
try:
    vs.plot_interpreter_image(inter)
except AssertionError:
    print("The heatmap is made of all zeros...")


# we now use another model: Saliency maps
inter2 = Interpreter(pipe.model, method="Saliency")

# interpret the mage
output = inter2.interpret(datum, class_)

# visualise the results
vs = Visualiser(pipe)
try:
    vs.plot_interpreter_image(inter2)
except AssertionError:
    print("The heatmap is made of all zeros...")


vs.plot_3d_dataset()

me = ModelExtractor(pipe.model, loss_fn)

list_of_layers = me.get_layers_param()

for k, item in list_of_layers.items():
    print(k, item.shape)

# the decision boundary will be available on tensorboard, in the projectors section.
x = next(iter(dl_tr))[0][0]
if x.dtype is not torch.int64:  # cannot backpropagate on integers!
    res = me.get_decision_boundary(x, n_epochs=1)
    res.shape

x = next(iter(dl_tr))[0]
list_activations = me.get_activations(x)
len(list_activations)

batch = next(iter(dl_tr))  # a whole batch!
if batch[0].dtype is torch.float:  # cannot backpropagate on integers!
    for gradient in me.get_gradients(batch)[1]:
        print(gradient.shape)

vs.plot_persistence_diagrams(batch)