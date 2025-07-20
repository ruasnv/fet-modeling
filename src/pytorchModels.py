# src/pytorchModels.py

import torch
from torch import nn


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 1)
        self.relu = nn.ReLU()  # ReLU activation

#TODO: Try with different activation functions later, GelU might be better for Cutoff region.

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.output_layer(x)