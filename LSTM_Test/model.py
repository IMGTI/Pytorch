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

        # CNN-LSTM

        import torch.nn.functional as F

        self.kernel_size = 1
        self.num_filters_cnn = 6

        self.conv1 = nn.Conv1d(1, self.num_filters_cnn, self.kernel_size)   # kernel size 1 <=> filter 1x1
        self.pool = nn.MaxPool1d(2, 2)

        self.lstm = nn.LSTM(input_size=self.num_filters_cnn, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)

        self.fc = nn.Linear(self.hidden_size*self.ways, self.num_classes)

    def forward(self, x, hidden=None):
        x = x.view(-1, 1, x.size()[-1])  # Change input shape for in_channels=1
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 5, self.num_filters_cnn) # Reshape output to show each filter output per sequence

        # Propagate input through LSTM
        if hidden:
            ula, hidden = self.lstm(x, hidden)
        else:
            ula, hidden = self.lstm(x)

        hidden = (hidden[0].detach(), hidden[1].detach())

        out = self.fc(ula[:,-1,:])

        return out, hidden

'''
        # LSTM

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
'''
