import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
from model import LSTM

class Train(object):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout,
                 state_dict_path, current, params_name):
        # Initialize the model
        self.lstm = LSTM(num_classes, input_size, hidden_size, num_layers, dropout)
        # Path to state dictionary
        self.state_dict_path = state_dict_path
        # Path and name for plots
        self.current = current
        self.params_name = params_name

        # Send net to GPU if possible
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pass

    def train_model(self, batch_size, learning_rate, num_epochs, times, defsX, defsY):
        # Send model to device
        self.lstm.to(self.device)

        self.criterion = torch.nn.MSELoss()    # mean-squared error for regression
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=learning_rate)

        # Try to load the model and optimizer 's state dictionaries
        self.load_model()

        # Train the model
        fig_loss = plt.figure(2)
        loss4plot = []

        if batch_size==-1:
            for epoch in tqdm(range(num_epochs), total=num_epochs):
                self.optimizer.zero_grad()

                outputs = self.lstm(defsX.to(self.device))

                # Obtain the value for the loss function
                loss = self.criterion(outputs.to(self.device), defsY.to(self.device))

                loss.backward()

                self.optimizer.step()

                loss4plot.append(loss)

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
            for epoch in tqdm(range(num_epochs), total=num_epochs):
                for batch in batches:
                    self.optimizer.zero_grad()

                    outputs = self.lstm(batch['defsX'].to(self.device))

                    # Obtain the value for the loss function
                    loss = self.criterion(outputs.to(self.device), batch['defsY'].to(self.device))

                    loss.backward()

                    self.optimizer.step()

                loss4plot.append(loss)

                # Plot loss vs epoch
                self.plot_loss(fig_loss, epoch, loss4plot)

                # Save model
                self.save_model(epoch, loss)
        pass

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

    def plot_loss(self, fig, epoch, loss4plot):
        fig_loss = fig
        fig_loss.clf()
        plt.plot(range(epoch+1), loss4plot, 'g-')
        plt.title("Mean running loss vs epoch")
        plt.xlabel("Epoch (units)")
        plt.ylabel("Running loss")
        fig_loss.savefig(self.current + "/loss_vs_epoch" + self.params_name + ".jpg")
        pass
