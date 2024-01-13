"""Some models to classify MNIST images"""

import torch
from torch import nn

class Linear(nn.Module):
    """Simple linear model."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10, bias=False)
    def forward(self, x):
        return self.fc(x)
    
class LinearHidden(nn.Module):
    """Linear model with a hidden layer."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64, bias=False)
        self.fc2 = nn.Linear(64, 10, bias=False)
    def forward(self, x):
        return self.fc2(self.fc1(x))
       
class CNN(nn.Module):
    """CNN with a large kernel and a global max or average pooling."""
    def __init__(self, max_pool=True):
        super().__init__()
        self.max_pool = max_pool

        self.conv = nn.Conv2d(1, 64, kernel_size=13, padding=13//2, bias=False)
        with torch.no_grad():
            self.conv.weight[:] = 0.
        if max_pool:
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        return self.forward_train(x)['out']

    def forward_train(self, x):
        """This method can be used for extracting all intermediate features."""
        maps = self.conv(x)
        pooled = self.pool(maps)
        pooled = torch.flatten(pooled, 1)
        out = self.fc(pooled)

        ret = {
            'maps':maps,
            'pooled':pooled,
            'out':out,
        }

        return ret

class CNN_old(nn.Module):
    """Deprecated model."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=13, padding=13//2, bias=False)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(64, 10, bias=False)
    def forward(self, x):
        x = self.pool(self.conv(x))
        x = torch.flatten(x, 1)
        return self.fc(x)
    
class CNN2_old(nn.Module):
    """Deprecated model."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=13, padding=13//2, bias=False)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(64, 10, bias=False)
    def forward(self, x):
        x = self.pool(self.conv(x))
        pooled_acts = torch.flatten(x, 1)
        return self.fc(pooled_acts), pooled_acts