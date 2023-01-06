import torch
import torch.nn as nn
import pandas as pd

class Net(nn.Module):
    def __init__(self, in_features=23):
        super().__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
