import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):

    def __init__(self, nodes_layer_1 = 8, dropout_p=0.0):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, nodes_layer_1, bias=True)
        self.drop_layer = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(nodes_layer_1, 2, bias=True)

        # Initialize weights to zero
        self.fc1.weight.data.fill_(0.)
        self.fc2.weight.data.fill_(0.)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.leaky_relu(self.fc2(x))
        return x


class DeeperNN(nn.Module):

    def __init__(self, dropout_p=0.0):
        super(DeeperNN, self).__init__()
        self.fc1 = nn.Linear(2, 16, bias=True)
        self.drop_layer = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(16, 32, bias=True)
        self.drop_layer = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(32, 64, bias=True)
        self.drop_layer = nn.Dropout(p=dropout_p)
        self.fc4 = nn.Linear(64, 2, bias=True)

        # Initialize weights to zero
        self.fc1.weight.data.fill_(0.)
        self.fc2.weight.data.fill_(0.)
        self.fc3.weight.data.fill_(0.)
        self.fc4.weight.data.fill_(0.)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.leaky_relu(self.fc2(x))
        return x