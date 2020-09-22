import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import os
from tqdm import tqdm

### HYPERPARAMETERS ###

# Net parameters
num_epochs = 2000#2000#300#2000
learning_rate = 0.001#0.001#0.01

input_size = 1
batch_size = 1  # Unused variable
hidden_size = 2
num_layers = 1

num_classes = 1


# Data parameters
seq_length = 100#4  # Train Window

train_size = -1000#int(len(y) * 0.67)
test_size = -1000#len(y) - train_size  # Unused variable


# Parameters in name for .jpg files
params_name = ('_e' + str(num_epochs) +
               '_lr' + str(learning_rate) +
               '_b' + str(batch_size) +
               '_i' + str(input_size) +
               '_n' + str(num_layers) +
               '_h' + str(hidden_size) +
               '_o' + str(num_classes) +
               '_trw' + str(seq_length))

# Create directory for each run and different hyperparameters

current = dt.datetime.now().strftime("%d_%m_%Y") + '/' + dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

# Create new directory for each run

# Create directory
try:
    os.mkdir(dt.datetime.now().strftime("%d_%m_%Y"))
    os.mkdir(current)
except:
    try:
        os.mkdir(current)
    except:
        pass

### SELECT DEVICE ###

# Send net to GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### DATA EXTRACT ###

file = 'Figura de Control.xlsx'
fig_name = 'F6'

data = pd.read_excel(file, fig_name, usecols=[0,1], names=['times', 'defs'])

try:
    times = np.array([dt.datetime.timestamp(x) for x in data['times']])
except:
    times = np.array(data['times'])
defs = np.array(data['defs'])

# Reshape data array from 1D to 2D
defs = defs.reshape(-1, 1)


### DATA PLOT ###

training_set = defs

fig1 = plt.figure(1)
fig1.clf()
plt.plot(training_set, 'r-', label = 'Raw Data')
plt.title('Deformation vs Time')
plt.ylabel('Defs(cm)')
plt.xlabel('Time(d)')
plt.grid(True)
plt.legend()
fig1.savefig(current + "/defs_vs_times" + params_name + ".jpg")

### DATA LOADING ###

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)

x, y = sliding_windows(training_data, seq_length)

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

### MODEL ###

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0.to(device), c_0.to(device)))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


### TRAINNING ####

# Initiate model

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
lstm.to(device)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
fig_loss = plt.figure(3)
loss4plot = []
for epoch in tqdm(range(num_epochs), total=num_epochs):
    outputs = lstm(trainX.to(device))
    optimizer.zero_grad()

    # obtain the loss function
    loss = criterion(outputs.to(device), trainY.to(device))

    loss.backward()

    optimizer.step()
    #if epoch % 100 == 0:
    #  print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    mean_loss = loss

    loss4plot.append(mean_loss)

    fig_loss.clf()
    plt.plot(range(num_epochs)[0:epoch+1], loss4plot, 'g-')
    plt.title("Mean running loss vs epoch")
    plt.xlabel("Epoch (units)")
    plt.ylabel("Running loss")
    fig_loss.savefig(current + "/loss_vs_epoch" + params_name + ".jpg")

### TESTING ###

lstm.eval()
train_predict = lstm(dataX.to(device))

data_predict = train_predict.data.cpu().numpy()
dataY_plot = dataY.data.cpu().numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

fig2 = plt.figure(2)
fig2.clf()

vline_substraction = np.absolute(train_size) - (seq_length + 1)
plt.axvline(x=len(y) - vline_substraction, c='r', linestyle='--')

plt.plot(dataY_plot, 'r-', label = 'Raw Data')
plt.plot(data_predict, 'g-', label = 'Predicted Data')
plt.title('Deformation vs Time')
plt.ylabel('Defs(cm)')
plt.xlabel('Time(d)')
plt.grid(True)
plt.legend()
fig2.savefig(current + "/defs_vs_times_pred" + params_name + ".jpg")
