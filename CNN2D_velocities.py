import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.utils.data as utils
import time

import pdb

torch.manual_seed(1)

class CNN2D_velocities(nn.Module):
    def __init__(self, n_input_channels=1, n_output=None):
        super().__init__()
              
        self.conv1 = nn.Conv2d(n_input_channels,3,5,padding=2)
        self.conv1_bn = nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True)
     
        self.fc1 = nn.Linear(10 * 164 * 3, 10 * 2 * 1)
            
    def forward(self, x):
        
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = x.view(-1, 10 * 164 * 3) # in order to reshape the tensor for as many columns we need
        x = self.fc1(x)
 
        return x
      
    def predict(self, x):
        return self.forward(x)