import torch.nn as nn
import torch
import torch.nn.functional as F

class MyModel(nn.Module):

    def __init__(self, n_features, n_hidden, n_outputs):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(n_hidden, n_outputs)
    
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


class MyModel2(nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_outputs)
        )

    def forward(self, x):
        return self.mlp(x)


# A simple model
class SimpleLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 15)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), -1)
        return x
