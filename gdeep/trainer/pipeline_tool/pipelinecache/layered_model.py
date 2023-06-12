import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributed.pipeline.sync.skip import stash, pop, skippable 

class x_layer(nn.Module):
    def forward(self, input):
        ret = input
        return input

class conv1_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
    def forward(self, input):
        ret = self.fc(input)
        return ret

class relu_layer(nn.Module):
    def forward(self, input):
        ret = torch.nn.functional.relu(input, inplace=False)
        return ret

class pool_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    def forward(self, input):
        ret = self.fc(input)
        return ret

class conv2_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    def forward(self, input):
        ret = self.fc(input)
        return ret

class relu_1_layer(nn.Module):
    def forward(self, input):
        ret = torch.nn.functional.relu(input, inplace=False)
        return ret

class pool_1_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    def forward(self, input):
        input = input.clone()
        ret = self.fc(input)
        return ret

class flatten_layer(nn.Module):
    def forward(self, input):
        ret = torch.flatten(input, 1)
        return ret

class fc1_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=400, out_features=120, bias=True)
    def forward(self, input):
        ret = self.fc(input)
        return ret

class relu_2_layer(nn.Module):
    def forward(self, input):
        ret = torch.nn.functional.relu(input, inplace=False)
        return ret

class fc2_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=120, out_features=84, bias=True)
    def forward(self, input):
        ret = self.fc(input)
        return ret

class relu_3_layer(nn.Module):
    def forward(self, input):
        ret = torch.nn.functional.relu(input, inplace=False)
        return ret

class fc3_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=84, out_features=10, bias=True)
    def forward(self, input):
        ret = self.fc(input)
        return ret

class output_layer(nn.Module):
    def forward(self, input):
        ret = input
        return ret

class PipelinedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s0 = nn.Sequential(x_layer(), conv1_layer(), relu_layer(), pool_layer(), conv2_layer(), relu_1_layer(), pool_1_layer()).cuda(0)
        self.s1 = nn.Sequential(flatten_layer(), fc1_layer(), relu_2_layer(), fc2_layer(), relu_3_layer(), fc3_layer(), output_layer()).cuda(1)
    def forward(self, input):
        ret = input
        ret = self.s0(ret.to(0))
        ret = self.s1(ret.to(1))
        return ret
    def get_modules(self):
        return  nn.Sequential(*[nn.Sequential(*self.s0),nn.Sequential(*self.s1)])
