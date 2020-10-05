import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from model import LSTM
from main import (num_classes, input_size, hidden_size,
                  num_layers, state_dict_path, dropout,
                  params_name, current, device)

class Train(self):
    def __init__(self):
        pass

    def train_model(self):
        # Initialize the model
        lstm = LSTM(num_classes, input_size, hidden_size, num_layers, dropout)

        # Try to load the model
        self.load_model()

        # Send model to device
        lstm.to(device)

        criterion = torch.nn.MSELoss()    # mean-squared error for regression
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

        # Train the model
        fig_loss = plt.figure(2)
        loss4plot = []
        for epoch in tqdm(range(num_epochs), total=num_epochs):
            outputs = lstm(trainX.to(device))
            optimizer.zero_grad()

            # Obtain the value for the loss function
            loss = criterion(outputs.to(device), trainY.to(device))

            loss.backward()

            optimizer.step()

            loss4plot.append(loss)

            # Plot loss vs epoch
            self.plot_loss(fig_loss, epoch, loss4plot)

            # Save model
            self.save_model()
        pass

    def save_model(self):
        # Save state dict of model
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': lstm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, state_dict_path)
        pass

    def load_model(self):
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
        pass

    def plot_loss(self, figure, epoch, loss4plot):
        fig_loss = figure
        fig_loss.clf()
        plt.plot(range(epoch+1), loss4plot, 'g-')
        plt.title("Mean running loss vs epoch")
        plt.xlabel("Epoch (units)")
        plt.ylabel("Running loss")
        fig_loss.savefig(current + "/loss_vs_epoch" + params_name + ".jpg")
        pass
