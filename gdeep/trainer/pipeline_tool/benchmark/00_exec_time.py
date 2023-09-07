import utils
import time
import subprocess

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from pathlib import Path

import sys
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29600'

# Ajoutez le chemin du projet à sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from pipeline_tool import SkippableTracing
from torch.distributed.pipeline.sync import Pipe 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_Caltech = utils.CaltechResnet()
model_orbits = None

dir_path = Path(__file__).resolve().parent / "training_launcher.py"
p = subprocess.run(['python', dir_path,
                            "exec_time",
                            "CNN",
                            "API torch",
                            '--gpu', str(1),
                            '--chunk', str(2),
                            '--epochs', str(2)], capture_output=True, text=True)

# ------ CNN ----------
print("Starting benchmark with CNN")

model_CNN = utils.Net()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_CNN.parameters(), lr=0.001, momentum=0.9)

model_CNN = model_CNN.to(device)

utils.capture_time(0, 2, model_CNN, trainloader, device, optimizer, criterion, 0)



# for input, label in trainloader:
#     input_size = input.shape
#     output_size = label.shape
#     break

# # With 1 GPU

# trace = SkippableTracing(1, utils.Net(), input_size, output_size)

# torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

# model_CNN = trace.get_modules()
# model_CNN = Pipe(model_CNN, chunks=2)

# utils.capture_time(1, 2, model_CNN, trainloader, device, optimizer, criterion, 1)
# torch.distributed.rpc.shutdown()
# # With 2 GPU

# trace = SkippableTracing(2, utils.Net(), input_size, output_size)

# torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

# model_CNN = trace.get_modules()
# model_CNN = Pipe(model_CNN, chunks=2)

# utils.capture_time(1, 2, model_CNN, trainloader, device, optimizer, criterion, 2)
# torch.distributed.rpc.shutdown()
# # ------ Caltech ------


# # ------ 