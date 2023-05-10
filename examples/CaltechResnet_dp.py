import CaltechResnet
import torch
from gdeep.data.datasets import DatasetBuilder, DataLoaderBuilder
from gdeep.trainer.trainer import Trainer, Parallelism, ParallelismType
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.sgd import SGD
from torch import nn

import os

if __name__ == '__main__':

    print(f'from base script: {os.getpid()}')

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
        {"sampler": samplers[0], "batch_size": 4},
        {"sampler": samplers[1], "batch_size": 1},
        {"sampler": samplers[2], "batch_size": 1}])

    # Create Resnet

    model = CaltechResnet.CaltechResnet()

    # Train resnet

    writer = SummaryWriter()
    loss_fn = nn.CrossEntropyLoss()
    train = Trainer(model, (caltech_dl_tr, caltech_dl_val, caltech_dl_ts), loss_fn, writer)
    devices = list(range(torch.cuda.device_count()))
    train.train(SGD, 1, False, {"lr": 0.001, "momentum": 0.9}, parallel=Parallelism(ParallelismType.FSDP_ZERO2, devices, len(devices)))

    # Test resnet to check performance

    acc, loss, cm = train.evaluate_classification(256, caltech_dl_ts)

    print(f"accuracy = {acc}%")
    print(f"loss = {loss}")
    #print(cm)
