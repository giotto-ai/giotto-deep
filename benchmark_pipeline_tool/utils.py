import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet50, resnet18,  ResNet50_Weights, ResNet18_Weights
import time

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

class TinyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    
class CaltechResnet(nn.Module):
    def __init__(self):
        super(CaltechResnet, self).__init__()
        self.flow = nn.Sequential(
            resnet18(weights=ResNet18_Weights.DEFAULT),
            nn.Linear(1000, 256)
        )

    def forward(self, x):
        return self.flow(x)
    

def training_normal(model, trainloader, device, optimizer, loss_fn):
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
    

def training_pipeline(model, trainloader, nb_gpu, optimizer, loss_fn):
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

def capture_time(mode, nb_epoch, model, trainloader, device, optimizer, loss_fn, nb_gpu):

    for epoch in range(nb_epoch):
        start_time = time.time()
        
        if mode == 0: # NORMAL
            training_normal(model, trainloader, device, optimizer, loss_fn)
        elif mode == 1: #PIPELINE
            training_pipeline(model, trainloader, nb_gpu, optimizer, loss_fn)

        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Execition time: {execution_time:.2f} seconds")
    
