import CaltechResnet
import torch
import argparse
from gdeep.data.datasets import DatasetBuilder, DataLoaderBuilder
from gdeep.trainer.trainer import Trainer, Parallelism, ParallelismType
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.sgd import SGD
from torch import nn

import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Giotto FSDP Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--n_epochs', type=int, default=1, metavar='N',
                        help='Number of epochs to train for (default: 1)')
    parser.add_argument('--fsdp', action='store_true', default=False,
                        help='Enable FSDP for training')
    parser.add_argument('--cv', action='store_true', default=False,
                        help='Enable cross-validation')
    parser.add_argument('--profiling', action='store_true', default=False,
                        help='Enable cross-validation')
    args = parser.parse_args()

    print(f'from base script: {os.getpid()}')

    # Recover Caltech256 dataset

    caltech_bd = DatasetBuilder("Caltech256")

    caltech_ds, _, _ = caltech_bd.build(transform=CaltechResnet.CALTECH_IMG_TRANSFORM, download=True)

    # Split dataset into training and test sets
    
    split_ratios = [1.] if args.cv else [0.8, 0.1]

    samplers = CaltechResnet.caltechSamplers(
        caltech_ds, split_ratios
    )

    # Create dataloaders

    if args.cv:
        print("Creating dataloaders for CV")
        caltech_dl_bd = DataLoaderBuilder((caltech_ds,))

        caltech_dl_tr, _, _ = caltech_dl_bd.build([
            {"sampler": samplers[0],"batch_size": args.batch_size},])

    else:
        print("Creating dataloaders")
        caltech_dl_bd = DataLoaderBuilder((caltech_ds, caltech_ds, caltech_ds))

        caltech_dl_tr, caltech_dl_ts, caltech_dl_val = caltech_dl_bd.build([
            {"sampler": samplers[0], "batch_size": args.batch_size},
            {"sampler": samplers[1], "batch_size": args.batch_size},
            {"sampler": samplers[2], "batch_size": args.batch_size}])

    # Create Resnet

    model = CaltechResnet.CaltechResnet()

    # Train resnet

    writer = SummaryWriter()
    loss_fn = nn.CrossEntropyLoss()
    dataloaders = (caltech_dl_tr,) if args.cv else (caltech_dl_tr, caltech_dl_val, caltech_dl_ts)
    train = Trainer(model, dataloaders, loss_fn, writer, print_every=20)
    devices = list(range(torch.cuda.device_count()))
    valloss, valacc = train.train(SGD, 
                                  args.n_epochs, 
                                  args.cv, 
                                  {"lr": 0.001, "momentum": 0.9}, 
                                  profiling=args.profiling, 
                                  parallel=Parallelism(ParallelismType.FSDP_ZERO2, devices, len(devices)) if args.fsdp else None)

    print(f'Training done: loss={valloss}, accuracy={valacc}')

    # Test resnet to check performance (if not CV)
    if not args.cv:
        acc, loss, cm = train.evaluate_classification(256, caltech_dl_ts)

        print(f"accuracy = {acc}%")
        print(f"loss = {loss}")
    #print(cm)
