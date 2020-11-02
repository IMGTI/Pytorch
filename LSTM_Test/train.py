import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
from model import LSTM

class Train(object):
    def __init__(self, batch_size, num_classes, input_size, hidden_size, num_layers, dropout,
                 bidirectional, state_dict_path, current, params_name, stateful=False):
        # Initialize the model
        self.lstm = LSTM(batch_size, num_classes, input_size, hidden_size, num_layers,
                         dropout, bidirectional)
        # Path to state dictionary
        self.state_dict_path = state_dict_path
        # Path and name for plots
        self.current = current
        self.params_name = params_name
        self.stateful = stateful

        # Send net to GPU if possible
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pass

    def train_model(self, batch_size, learning_rate, num_epochs, times, defsX, defsY,
                    validate=True):
        # Define validation set and training set
        if validate:
            defsX = defsX[:-10]
            defsY = defsY[:-10]
            val_defsX = defsX[-10:]
            val_defsY = defsY[-10:]

        # Send model to device
        self.lstm.to(self.device)

        self.criterion = torch.nn.MSELoss()    # mean-squared error for regression
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=learning_rate)

        # Try to load the model and optimizer 's state dictionaries
        self.load_model()

        # Train the model
        fig_loss = plt.figure(2)
        loss4plot = []
        val_loss4plot = []

        if batch_size==-1:
            for epoch in tqdm(range(num_epochs), total=num_epochs):
                self.optimizer.zero_grad()

                outputs, hidden = self.lstm(defsX.to(self.device))

                # Obtain the value for the loss function
                loss = self.criterion(outputs.to(self.device), defsY.to(self.device))

                loss.backward()

                self.optimizer.step()

                loss4plot.append(loss.item())

                # Plot loss vs epoch
                self.plot_loss(fig_loss, epoch, loss4plot)

                # Save model
                self.save_model(epoch, loss)
        else:
            batches = []
            ind = 0
            while True:
                try:
                    batches.append({'defsX':torch.index_select(defsX, 0, torch.tensor(np.int64(np.arange(ind,ind+batch_size,1)))),
                                    'defsY':torch.index_select(defsY, 0, torch.tensor(np.int64(np.arange(ind,ind+batch_size,1))))})
                    ind += batch_size
                except:
                    break

            if (batches[-1]['defsX']).size(0)!=batch_size:
                batches = batches[:-1]
                print("Removing last batch because of invalid batch size")

            for epoch in tqdm(range(num_epochs), total=num_epochs):
                hidden = None
                for batch in batches:
                    self.optimizer.zero_grad()

                    if self.stateful:
                        outputs, hidden = self.lstm(batch['defsX'].to(self.device), hidden=hidden)
                    else:
                        outputs, hidden = self.lstm(batch['defsX'].to(self.device))

                    # Obtain the value for the loss function
                    loss = self.criterion(outputs.to(self.device), batch['defsY'].to(self.device))

                    loss.backward()

                    self.optimizer.step()

                    with torch.no_grad():
                        # Initialize model in testing mode
                        self.lstm.eval()
                        val_pred, val_hidden = self.lstm(val_defsX.to(self.device))
                        val_loss = self.criterion(val_pred.to(self.device), val_defsY.to(self.device))
                        # Initialize model in trainning mode again
                        self.lstm.train()

                loss4plot.append(loss.item())
                val_loss4plot.append(val_loss.item())

                # Plot loss vs epoch
                self.plot_loss(fig_loss, epoch, loss4plot, val_loss4plot)

                # Save model
                self.save_model(epoch, loss)
        pass

        '''
        val_inputs = np.zeros([np.absolute(train_size) + 1, 1, batch['defsX'].size(1), 1])

        val_inputs[0] = batch['defsX'].reshape(1,seq_length,1).data.numpy()

        time_step = np.absolute(times[0] - times[1])

        self.times = times
        self.times_predictions = (np.arange(0, (fut_pred+1)*time_step, time_step) +
                                  times[ind_test])

        for i in range(fut_pred):
            seq = torch.FloatTensor(test_inputs[i]).to(self.device)
            with torch.no_grad():
                prediction, hidden = self.lstm(seq)
                prediction = prediction.data.cpu().numpy().item()
                test_inputs[i+1] = np.append(test_inputs[i][0][1:], prediction).reshape([1,seq_length,1])


        data_predict = np.array([x.reshape(seq_length)[-1] for x in test_inputs]).reshape([-1,1])
        dataY_plot = defsY.data.cpu().numpy()

        data_predict = sc.inverse_transform(data_predict)
        dataY_plot = sc.inverse_transform(dataY_plot)
        '''

    def save_model(self, epoch, loss):
        # Save state dict of model
        torch.save({
                    #'epoch': epoch,
                    'model_state_dict': self.lstm.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    #'loss': loss,
                    }, self.state_dict_path)
        pass

    def load_model(self):
        # Load state dict of model
        try:
            checkpoint = torch.load(self.state_dict_path)
            self.lstm.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #epoch = checkpoint['epoch']
            #loss = checkpoint['loss']
            self.lstm.to(self.device)
            self.lstm.train()
        except:
            print('State dict(s) missing')
        pass

    def plot_loss(self, fig, epoch, loss4plot, val_loss4plot):
        fig_loss = fig
        fig_loss.clf()
        plt.plot(range(epoch+1), loss4plot, 'g-', label='Training loss')
        plt.plot(range(epoch+1), val_loss4plot, 'r-', label='Validation loss')
        plt.title("Mean running loss vs epoch")
        plt.xlabel("Epoch (units)")
        plt.ylabel("Running loss")
        plt.grid(True)
        plt.legend()
        fig_loss.savefig(self.current + "/loss_vs_epoch" + self.params_name + ".jpg")
        pass
