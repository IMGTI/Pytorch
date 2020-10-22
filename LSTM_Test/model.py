import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout, bidirectional):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        if self.bidirectional:
            self.ways = 2
        else:
            self.ways = 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)

        self.fc = nn.Linear(self.hidden_size*self.ways, self.num_classes)

        # Send net to GPU if possible
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pass

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers*self.ways, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers*self.ways, x.size(0), self.hidden_size))
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0.to(self.device), c_0.to(self.device)))
        
        out = self.fc(ula[:,-1,:])

        return out
