import subprocess
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import argparse

import torchvision.transforms as transforms
import torchvision

import argparse
import utils

import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29600'

import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from pipeline_tool import SkippableTracing
from torch.distributed.pipeline.sync import Pipe 

# Créez un objet ArgumentParser
parser = argparse.ArgumentParser(description="Script d'analyse avec différentes options")

parser.add_argument(
    "mode",
    choices=["memory_alloc", "exec_time", "accuracy"],
    help="Mode d'analyse (memory_alloc, exec_time, accuracy)"
)
parser.add_argument(
    "model",
    choices=["CNN", "Resnet18", "Resnet50", "orbit5k"],
    help="Modèle à utiliser (CNN, Resnet18, Resnet50, orbit5k)"
)
parser.add_argument(
    "framework",
    choices=["Pipeline", "API torch"],
    help="Choix du framework (Pipeline, API torch)"
)
parser.add_argument(
    "--gpu",
    type=int,
    default=1,
    help="Nombre de GPU à utiliser (par défaut 1)"
)
parser.add_argument(
    "--chunk",
    type=int,
    default=10,
    help="Nombre de chunks (par défaut 10)"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Nombre d'époques (par défaut 100)"
)

args = parser.parse_args()

if args.model == "CNN":
    model = utils.Net()

elif args.model == "Resnet18":
    model = utils.CaltechResnet()

elif args.model == "Resnet50":
    pass

elif args.model == "orbit5k":
    pass

else: 
    raise ValueError("Given model is not known.")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

if args.framework == "API torch":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            input, label = data
            input, label = input.to(device), label.to(device)

            optimizer.zero_grad()

            output = model(input)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()


elif args.framework == "Pipeline":
    nb_gpu = args.gpu
    chunk = args.chunk

    for input, label in trainloader:
        input_size = input.shape
        output_size = label.shape
        break

    trace = SkippableTracing(nb_gpu, model, input_size, output_size)
    torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
    model = trace.get_modules()
    model = Pipe(model, chunks=chunk)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            input, label = data
            input = input.to(0)
            label = label.to(nb_gpu - 1)

            optimizer.zero_grad()

            output = model(input).local_value()

            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

else:
    raise ValueError("Framework specified is not known.")