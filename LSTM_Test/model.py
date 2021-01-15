import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F

class LSTM(nn.Module):
    def __init__(self, batch_size, num_classes, input_size, hidden_size, num_layers, dropout,
                 bidirectional, seed):
        # RNG Seed
        np.random.seed(seed)  # Numpy
        torch.manual_seed(seed)  # Pytorch

        # Inherit nn.Module class
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


class CNNLSTM(nn.Module):
    def __init__(self, batch_size, num_classes, input_size, hidden_size, num_layers, dropout,
                 bidirectional, filters_number, kernel_size, seed):
        # RNG Seed
        np.random.seed(seed)  # Numpy
        torch.manual_seed(seed)  # Pytorch

        # Inherit nn.Module class
        super(CNNLSTM, self).__init__()

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

        self.kernel_size = kernel_size
        self.filters_number = filters_number

        self.conv1 = nn.Conv1d(1, self.filters_number, self.kernel_size)   # kernel size 1 <=> filter 1x1
        self.conv2 = nn.Conv1d(self.filters_number, self.filters_number, self.kernel_size)
        self.conv3 = nn.Conv1d(self.filters_number, self.filters_number, self.kernel_size)
        #self.pool = nn.MaxPool1d(2, 2)

        self.lstm = nn.LSTM(input_size=self.filters_number, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)

        self.fc = nn.Linear(self.hidden_size*self.ways, self.num_classes)

    def forward(self, x, hidden=None):
        x = x.view(-1, 1, x.size()[1])  # Change input shape for in_channels=1
                                        # x.size()[1]==window_size
        #x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.transpose(x, 1, 2) # Reshape output to show each filter output per sequence
                                     # because LSTM needs that input shape

        # Propagate input through LSTM
        if hidden:
            ula, hidden = self.lstm(x, hidden)
        else:
            ula, hidden = self.lstm(x)

        hidden = (hidden[0].detach(), hidden[1].detach())

        out = self.fc(ula[:,-1,:])

        return out, hidden
