import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self, input_size, num_classes, filters_number, kernel_size, seed):
        # RNG Seed
        np.random.seed(seed)  # Numpy
        torch.manual_seed(seed)  # Pytorch

        # Inherit nn.Module class
        super(CNN, self).__init__()

        self.input_size = input_size  # Number of channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.filters_number = filters_number

        # CNN

        self.conv1 = nn.Conv1d(self.input_size, self.filters_number, self.kernel_size)   # kernel size 1 <=> filter 1x1
        self.conv2 = nn.Conv1d(self.filters_number, self.filters_number, self.kernel_size)
        self.conv3 = nn.Conv1d(self.filters_number, self.filters_number, self.kernel_size)

        self.pool = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(self.filters_number*268, 32)  # (2151 - (ks-stride))/2
                                                           # (stride=1, 3 times, one for each conv+pool)
        self.fc2 = nn.Linear(32, self.num_classes)

    def forward(self, x, hidden=None):
        x = x.view(-1, 1, x.size()[1])  # Change input shape for in_channels=1
                                        # x.size()[1]==window_size

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1,self.filters_number*268)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
