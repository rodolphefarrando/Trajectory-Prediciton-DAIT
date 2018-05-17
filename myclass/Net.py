from torch import nn
from torch import optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.utils.data as utils

class Net(torch.nn.Module):
    def __init__(self, n_feature,n_hidden, n_output):
        super(Net, self).__init__()

        # Defining the layers, 128, 64, 10 units each
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)


    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x