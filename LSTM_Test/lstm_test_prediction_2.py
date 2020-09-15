import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from tqdm import tqdm

# Moving average
def smooth(data, N_avg):
    def mov_avg(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    return mov_avg(data, N_avg)

# Hyperparameters
learning_rate = 0.1#0.001
epochs = 2#10#20#150#2#10#150

batch_size = 1#1
input_size = 1#1
num_layers = 3#10
hidden_layer_size = 6#100
output_size = 1#1

test_data_size = 100#1000#12#100  # 1 refers to -1 index of dataset, ie.,
                         # whole dataset is training

train_window = 1000#100#1000#12  # Ventanas de tiempo usadas para crear las secuencias

                               # 12 datos equivalen a 1 hora
fut_pred = 100#1000#100#12


# Get flight dataset from seaborn
#flight_data = sns.load_dataset("flights")
#flight_data.head()

#print(flight_data.columns)

#file = 'prueba_serie.xlsx'
#fig_name = 'Sheet1'
file = 'Figura_de_control_desde_feb.xlsx'
fig_name = 'Datos'

data = pd.read_excel(file, fig_name, usecols=[0,1], names=['times', 'defs'])

try:
    times = np.array([dt.datetime.timestamp(x) for x in data['times']])
except:
    times = np.array(data['times'])
defs = np.array(data['defs'])

N_avg = 2#5#2  # 2 para hacer una linea recta (tendencia) y al menos
               # 5 puntos para tendencia valida (entonces con N_avg=2
               # se logran 2-3 smooth ptos por cada 5)
times = smooth(times, N_avg)
defs = smooth(defs, N_avg)

### Data processing

# Change data to float
#all_data = flight_data['passengers'].values.astype(float)
times = times.astype(float)
defs = defs.astype(float)

# Divide data in train and test set
#test_data_size = 12

#train_data = all_data[:-test_data_size]
#test_data = all_data[-test_data_size:]
train_data = defs[:-test_data_size]
test_data = defs[-test_data_size:]

#print(len(train_data))
#print(len(test_data))

# Normalize the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

#print(train_data_normalized[:5])
#print(train_data_normalized[-5:])

# Convert data into tensors
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

#train_window = 12  # Ventanas de tiempo usadas para crear las secuencias

# Create sequences
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]  # Usa como label de la secuencia
                                               # el siguiente
                                               # valor de la serie de tiempo,
                                               # ie, el valor a predecir de
                                               # esa secuencia
        inout_seq.append((train_seq ,train_label))
    return inout_seq

# Create data sequences with corresponding labels
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

### Create LSTM

class LSTM(nn.Module):
    def __init__(self, input_size=input_size, hidden_layer_size=hidden_layer_size,
                 output_size=output_size, num_layers=num_layers, batch_size=batch_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(num_layers, batch_size, self.hidden_layer_size),
                            torch.zeros(num_layers, batch_size, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

### Trainning

# Initiate the model
model = LSTM()

# Load state dict of model
try:
    model.load_state_dict(torch.load('state_dict_2'))
    model.eval()
except:
    print('No model state dict found')

# Send net to GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

fig_loss = plt.figure(3)
loss4plot = []

for i in tqdm(range(epochs), total=epochs):
    running_loss = 0.0
    for seq, labels in train_inout_seq:
        seq, labels = seq.to(device), labels.to(device)

        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(num_layers, batch_size, model.hidden_layer_size).to(device),
                             torch.zeros(num_layers, batch_size, model.hidden_layer_size).to(device))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()

        running_loss += single_loss.item()

        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    # Save state dict of model
    #torch.save(model.state_dict(), 'state_dict_2')

    # Plot loss vs epoch
    mean_loss = running_loss/len(train_inout_seq)

    loss4plot.append(mean_loss)

    fig_loss.clf()
    plt.plot(range(epochs)[0:i+1], loss4plot, 'g-')
    plt.title("Mean running loss vs epoch")
    plt.xlabel("Epoch (units)")
    plt.ylabel("Running loss")
    fig_loss.savefig("loss_vs_epoch_2.jpg")

#print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

### Test

# Prediction

# Use last #"train_window" elements from train_data as input
test_inputs = train_data_normalized[-train_window:].tolist()  # Number of inputs
                                                              # used, should be
                                                              # same as number
                                                              # of elements used
                                                              # for train window
#print(test_inputs)

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:]).to(device)
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())

#print(test_inputs[fut_pred:])

# Unnormalize the predictions
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
#print(actual_predictions)

### Plot predictions

# All test input
#x = np.arange(132, 144, 1)
#print(x)

times = (times/(3600*24) -
        (times/(3600*24))[0])

time_step = np.absolute(times[0] - times[1])

times_predictions = (np.arange(0, fut_pred*time_step, time_step) +
                     times[-test_data_size])

fig1 = plt.figure(1)
fig1.clf()
plt.title('Deformation vs Time')
plt.ylabel('Defs')
plt.xlabel('Time(d)')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
#plt.plot(flight_data['passengers'])
plt.plot(times[-200:], defs[-200:])
#plt.plot(x,actual_predictions)
plt.plot(times_predictions,actual_predictions)
fig1.savefig('defs_vs_times.jpg')

# Last 12 months
fig2 = plt.figure(2)
fig2.clf()
plt.title('Deformation vs Time')
plt.ylabel('Defs')
plt.xlabel('Time')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
#plt.plot(flight_data['passengers'][-train_window:])
plt.plot(times[-test_data_size:], defs[-test_data_size:])
#plt.plot(x,actual_predictions)
plt.plot(times_predictions,actual_predictions)
fig2.savefig('defs_vs_times_12times.jpg')
