import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from LSTM import *
from LSTMbis import *
import pandas as pd
import numpy as np
import torch.utils.data as utils
import time

import pdb

torch.manual_seed(1)


class CNN1D(nn.Module):
    def __init__(self, n_input_channels=1, n_output=None):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_input_channels,8,5,padding=2)
        self.conv2 = nn.Conv1d(8,30,3,padding=1)
        self.conv3 = nn.Conv1d(30,40,3,padding=1)
        self.conv3_bn = nn.BatchNorm1d(40, eps=1e-05, momentum=0.1, affine=True)
     
        self.fc1 = nn.Linear(9*2 * 40, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 18)

    
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = x.view(-1, 9*2 * 40) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def predict(self, x):
        return self.forward(x)