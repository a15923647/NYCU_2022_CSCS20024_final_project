import torch
import torch.nn as nn
import pandas as pd
'''
class Net(nn.Module):
  def __init__(self,input_shape=23):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape,32)
    self.fc2 = nn.Linear(32,64)
    self.fc3 = nn.Linear(64,1)
  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x
'''
'''
class Net(nn.Module):
    def __init__(self, in_features=23):
        super().__init__()
        self.fc = nn.Sequential(
            #nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 32),
            nn.ReLU(),
            #nn.BatchNorm1d(32),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            #nn.BatchNorm1d(128),
            #nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            #nn.BatchNorm1d(256),
            #nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
'''
'''
class Net(nn.Module):
    def __init__(self, in_features=23):
        super().__init__()
        self.fc = nn.Sequential(
            #nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 48),
            nn.ReLU(),
            nn.BatchNorm1d(48),
            nn.Dropout(),
            #nn.ReLU(),
            nn.Linear(48, 96),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            #nn.ReLU(),
            nn.Dropout(),
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            #nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
'''
'''
class Net(nn.Module):
    def __init__(self, in_features=23):
        super().__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
'''

class Net(nn.Module):
    def __init__(self, in_features=23):
        super().__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 24),
            nn.ReLU(),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
'''
class Net(nn.Module):
    def __init__(self, in_features=21):
        super().__init__()
        self.fc = nn.Sequential(
            #nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
'''