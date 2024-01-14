import numpy as np
import torch
import matplotlib.pyplot as plt

class ResNetSampler(torch.nn.Module):
    """Sample activations of each stage of a Pytorch's ResNet model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):

        model = self.model
        features = {}

        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        features['relu'] = x
        x = model.maxpool(x)

        x = model.layer1(x)
        features['layer1'] = x
        x = model.layer2(x)
        features['layer2'] = x
        x = model.layer3(x)
        features['layer3'] = x
        x = model.layer4(x)
        features['layer4'] = x

        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc(x)
        features['out'] = x

        return features

def plot_with_grid(img):
    """Plot image with grid lines."""

    plt.imshow(img, 'gray', interpolation='none')
    plt.vlines(np.arange(-0.5,img.shape[1]), -0.5, img.shape[0]-0.5, linewidth=0.5)
    plt.hlines(np.arange(-0.5,img.shape[0]), -0.5, img.shape[1]-0.5, linewidth=0.5)


