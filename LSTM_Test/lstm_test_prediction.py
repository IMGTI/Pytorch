import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm

### Hyperparameters

batch_size = 1
learning_rate = 0.1#0.001
num_epochs = 100
num_layers = 10
lstm_input_dim_size = 1
output_dim = 8842
hidden_layers_num = 1


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        # Only take the output from the final timestep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

model = LSTM(lstm_input_dim_size, hidden_layers_num, batch_size=batch_size, output_dim=output_dim, num_layers=num_layers)

### Load state dict of model
try:
    model.load_state_dict(torch.load('state_dict'))
    model.eval()
except:
    print('No model state dict found')

### SEND NET TO GPU IF AVAILABLE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

loss_fn = torch.nn.MSELoss(size_average=False)

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

file = 'Figura de Control.xlsx'
fig_name = 'F6'

data = pd.read_excel(file, fig_name, usecols=[0,1], names=['times', 'defs'])

x_train = np.array([dt.datetime.timestamp(x) for x in data['times']])
y_train = np.array(data['defs'])

#x_train = x_train.reshape(len(x_train),1)
#y_train = y_train.reshape(len(y_train),1)

x_train = torch.from_numpy(x_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)

hist = np.zeros(num_epochs)

fig_loss = plt.figure(2)
loss4plot = []

for ind_epoch, t in tqdm(enumerate(range(num_epochs)), total=num_epochs):
    ### Training only for one y_train

    # Clear stored gradient
    model.zero_grad()

    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()

    # Forward pass
    y_pred = model(x_train).to(device)

    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

    ### Save state dict of model
    torch.save(model.state_dict(), 'state_dict')

    loss4plot.append(loss)

    fig_loss.clf()
    plt.plot(range(num_epochs)[0:ind_epoch+1], loss4plot, 'g-')
    plt.title("Mean running loss vs epoch")
    plt.xlabel("Epoch (units)")
    plt.ylabel("Running loss")
    fig_loss.savefig("loss_vs_epoch.jpg")

### Plot results

time = (x_train.cpu().detach().numpy()/(3600*24) -
        (x_train.cpu().detach().numpy()/(3600*24))[0])

fig = plt.figure(1)
fig.clf()
plt.plot(time, y_train.cpu(), 'g-', label='Ground Truth')
plt.plot(time, y_pred.cpu().detach().numpy(), 'r-', label='Predicted')
plt.title('Deformation vs time')
plt.xlabel('Time (d)')
plt.ylabel('Deformation (cm)')
plt.legend()
fig.savefig('train_vs_predicted.jpg')
