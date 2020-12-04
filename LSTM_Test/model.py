import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, batch_size, num_classes, input_size, hidden_size, num_layers, dropout,
                 bidirectional, seed):
        # RNG Seed
        np.random.seed(seed)  # Numpy
        torch.manual_seed(seed)  # Pytorch

        # Inherit LSTM class
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_size = batch_size

        if self.bidirectional:
            self.ways = 2
        else:
            self.ways = 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)

        self.fc = nn.Linear(self.hidden_size*self.ways, self.num_classes)

    def forward(self, x, hidden=None):
        # Propagate input through LSTM
        if hidden:
            ula, hidden = self.lstm(x, hidden)
        else:
            ula, hidden = self.lstm(x)

        hidden = (hidden[0].detach(), hidden[1].detach())

        out = self.fc(ula[:,-1,:])

        return out, hidden
