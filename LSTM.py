import torch
from torch import nn
from math import sqrt
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        hidden_size = 64
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=4,
            hidden_size=hidden_size,         # rnn hidden unit

            num_layers=2,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(hidden_size, 4)
        #self.out = nn.Linear(128, 20)
        #self.out = nn.Linear(20, 2)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out)
        for i in range(10):
            if i == 0:
                out[i, :, 0:1] = x[-1, :, 0:1] + out[i, :, 2:3]
            else:
                out[i, :, 0:1] = out[i - 1, :, 0:1] + out[i, :, 2:3]
        return out


    def predict(self,x):
        x = self.forward(x)
        return x
