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

# Moving average
def smooth(data, N_avg):
    def mov_avg(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    return mov_avg(data, N_avg)

### HYPERPARAMETERS ###

# Net parameters
num_epochs = 2000#200#300#2000
learning_rate = 0.001#0.001#0.01

input_size = 1
batch_size = 1  # Unused variable
hidden_size = 100#10#2
num_layers = 1

num_classes = 1


# Data parameters
seq_length = 12#1000#4  # Train Window
                        # 1h = 12
                        # 5min = 1

train_size = -100#int(len(y) * 0.67)
test_size = -100#len(y) - train_size  # Unused variable

fut_pred = 12#100  # Number of predictions

drop = 0.05#0.05

# Parameters in name for .jpg files
params_name = ('_e' + str(num_epochs) +
               '_lr' + str(learning_rate) +
               '_b' + str(batch_size) +
               '_i' + str(input_size) +
               '_n' + str(num_layers) +
               '_h' + str(hidden_size) +
               '_o' + str(num_classes) +
               '_trw' + str(seq_length) +
               '_drp' + str(drop))

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

#file = 'Figura de Control.xlsx'
#fig_name = 'F6'
#file = 'prueba_serie.xlsx'
#fig_name = 'Sheet1'
file = 'Figura_de_control_desde_feb.xlsx'
fig_name = 'Datos'

data = pd.read_excel(file, fig_name, usecols=[0,1], names=['times', 'defs'])

try:
    times = np.array([dt.datetime.timestamp(x) for x in data['times']])
except:
    times = np.array(data['times'])

# Convert times from seconds to days
times = (times/(3600*24) -
        (times/(3600*24))[0])

defs = np.array(data['defs'])

# Apply smooth
N_avg = 2#5#2    # 2 para hacer una linea recta (tendencia) y al menos
               # 5 puntos para tendencia valida (entonces con N_avg=2
               # se logran 2-3 smooth ptos por cada 5)
times = smooth(times, N_avg)
defs = smooth(defs, N_avg)

# Reshape data array from 1D to 2D
defs = defs.reshape(-1, 1)

### DATA PLOT ###

training_set = defs

fig1 = plt.figure(1)
fig1.clf()
plt.plot(times, training_set, 'r-', label = 'Raw Data')
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
                            num_layers=num_layers, batch_first=True,
                            dropout=drop)

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

state_dict_path = 'state_dict'

# Initiate model

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

# Load state dict of model
try:
    checkpoint = torch.load(state_dict_path)
    lstm.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    lstm.train()
except:
    print('State dict(s) missing')

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

    # Save state dict of model
    torch.save({
                'epoch': epoch,
                'model_state_dict': lstm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, state_dict_path)

### TESTING ###

# Test predctions over time

lstm.eval()

test_inputs = np.zeros([fut_pred + 1, 1, seq_length, 1])

ind_test = -100#5000#1000#len(dataX)-1
#test_inputs[0] = dataX[-1].reshape(-1,seq_length,1).data.numpy()

test_inputs[0] = dataX[ind_test].reshape(-1,seq_length,1).data.numpy()

time_step = np.absolute(times[0] - times[1])

times_dataY = (times + (seq_length*time_step))[:-seq_length-1]  # Times according with dataX and dataY dimensions

times_predictions = (np.arange(0, (fut_pred+1)*time_step, time_step) +
                     times_dataY[ind_test])

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[i]).to(device)
    #print(seq)
    with torch.no_grad():
        prediction = lstm(seq).data.cpu().numpy().item()
        test_inputs[i+1] = np.append(test_inputs[i][0][1:], prediction).reshape([1,seq_length,1])

'''REVISAR PORQUE LSTM(SEQ) ES UN NUM CON LA PREDICCION DE SEQ,
POR LO QUE AL APPENDEAR AL FINAL DE TEST_INPUT, LUEGO NO LE ESTOY DANDO UNA
SECUENCIA (SEQ) A LA PROXIMA ITERACION, SINO UN NUMERO, Y EL TORCH.FLOATTENSOR
GENERA UN TENSOR CON EL MISMO VALOR EN TODOS LOS INDICES SI SE LE DA COMO ARGUMENTO
UN SOLO VALOR (DONE)'''

#print(test_inputs)

data_predict = np.array([x.reshape(seq_length)[-1] for x in test_inputs]).reshape([-1,1])
dataY_plot = dataY.data.cpu().numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

#print(data_predict)

fig2 = plt.figure(2)
fig2.clf()

#plt.plot(range(-train_size), dataY_plot[train_size:], 'r-', label = 'Raw Data')
plt.plot(times_dataY[ind_test:ind_test+(fut_pred+1)], dataY_plot[ind_test-1:ind_test-1+(fut_pred+1)], 'r-', label = 'Raw Data')
#plt.plot(range(len(data_predict)), data_predict, 'g-', label = 'Predicted Data')
plt.plot(times_predictions, data_predict, 'g-', label = 'Predicted Data')
if fut_pred>=seq_length:
    plt.axvline(x=times_predictions[seq_length-1], c='b', linestyle='--')
plt.title('Deformation vs Time')
plt.ylabel('Defs(cm)')
plt.xlabel('Time(d)')
plt.grid(True)
plt.legend()
fig2.savefig(current + "/defs_vs_times_pred" + params_name + ".jpg")
data_predict2 = data_predict
# Test fitting model

lstm.eval()

train_predict = lstm(dataX.to(device))  # Should be the same length as dataX
                                        # but in a delayed window by 1 time
                                        # (prediction) ==> last value

data_predict = train_predict.data.cpu().numpy()
dataY_plot = dataY.data.cpu().numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

fig3 = plt.figure(3)
fig3.clf()

#vline_substraction = np.absolute(train_size)# - (seq_length + 1)
vline_substraction = times_dataY[ind_test]
plt.axvline(x=vline_substraction, c='r', linestyle='--')

plt.plot(times_dataY, dataY_plot, 'r-', label = 'Raw Data')
plt.plot(times_dataY, data_predict, 'g-', label = 'Predicted Data')
plt.title('Deformation vs Time')
plt.ylabel('Defs(cm)')
plt.xlabel('Time(d)')
plt.grid(True)
plt.legend()
fig3.savefig(current + "/defs_vs_times_pred_fitting" + params_name + ".jpg")
