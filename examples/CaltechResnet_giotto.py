import CaltechResnet
from gdeep.data.datasets import DatasetBuilder, DataLoaderBuilder
from gdeep.trainer.trainer import Trainer
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.sgd import SGD
from torch import nn

# Recover Caltech256 dataset

caltech_bd = DatasetBuilder("Caltech256")

caltech_ds, _, _ = caltech_bd.build(transform=CaltechResnet.CALTECH_IMG_TRANSFORM, download=True)

# Split dataset into training and test sets
split_ratios = [0.8, 0.1]

samplers = CaltechResnet.caltechSamplers(
    caltech_ds, split_ratios
)

# Create dataloaders

caltech_dl_bd = DataLoaderBuilder((caltech_ds, caltech_ds, caltech_ds))

caltech_dl_tr, caltech_dl_ts, caltech_dl_val = caltech_dl_bd.build([
    {"sampler": samplers[0], "batch_size": 32},
    {"sampler": samplers[1], "batch_size": 16},
    {"sampler": samplers[2], "batch_size": 16}])

# Create Resnet

model = CaltechResnet.CaltechResnet()

# Train resnet

writer = SummaryWriter()
loss_fn = nn.CrossEntropyLoss()
train = Trainer(model, (caltech_dl_tr, caltech_dl_val, caltech_dl_ts), loss_fn, writer, print_every=20)
train.train(SGD, 1, False, {"lr": 0.001, "momentum": 0.9}, profiling=True)

# Test resnet to check performance

acc, loss, cm = train.evaluate_classification(256, caltech_dl_ts)

print(f"accuracy = {acc}%")
print(f"loss = {loss}")
#print(cm)
