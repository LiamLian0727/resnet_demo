import torch
import torchvision
import pandas as pd
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides) if use_1x1conv else None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    block = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            block.append(Residual(num_channels, num_channels))
    return block


class ResNet(nn.Module):
    def __init__(self, input_channels, block, num_class, layers_num=[2, 2, 2, 2]):
        super().__init__()
        self.transforms = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU())
        self.layers = nn.ModuleList()
        hidden_channels = 64
        for idx, i in enumerate(layers_num):
            out_channels = hidden_channels * 2 if bool(idx) else hidden_channels
            self.layers.append(
                nn.Sequential(*block(hidden_channels, out_channels, i, first_block=not bool(idx))))
            hidden_channels = hidden_channels * 2 if bool(idx) else hidden_channels
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(hidden_channels, num_class))

    def forward(self, X):
        X = self.transforms(X)
        for layer in self.layers:
            X = layer(X)
        return self.out(X)

def resnet_18(input_channels, num_class):
    return ResNet(input_channels, resnet_block, num_class, [2, 2, 2, 2])
