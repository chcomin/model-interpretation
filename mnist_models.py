import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10, bias=False)
    def forward(self, x):
        return self.fc(x)
    
class LinearHidden(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64, bias=False)
        self.fc2 = nn.Linear(64, 10, bias=False)
    def forward(self, x):
        return self.fc2(self.fc1(x))
       
class CNN_old(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=13, padding=13//2, bias=False)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(64, 10, bias=False)
    def forward(self, x):
        x = self.pool(self.conv(x))
        pooled_acts = torch.flatten(x, 1)
        return self.fc(pooled_acts), pooled_acts
    
class CNN(nn.Module):
    def __init__(self, gated=False, gate_at_pool=False):
        super().__init__()
        self.gated = gated
        self.gate_at_pool = gate_at_pool

        self.conv = nn.Conv2d(1, 64, kernel_size=13, padding=13//2, bias=False)
        with torch.no_grad():
            self.conv.weight[:] = 0.
        #if gated:
        #    self.att = nn.Parameter(torch.ones(64))
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        return self.forward_train(x)['out']

    def forward_train(self, x):
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