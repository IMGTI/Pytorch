import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
from model_spc import CNN
from pytorchtools import EarlyStopping

class Train(object):
    def __init__(self, batch_size, input_size, num_classes, filters_number, kernel_size,
                 state_dict_path, current, params_name, seed):
        # RNG Seed
        np.random.seed(seed)  # Numpy
        torch.manual_seed(seed)  # Pytorch

        # Initialize the model
        self.cnn =  CNN(input_size, num_classes, filters_number, kernel_size, seed)

        # Print model summary
        #print("Number of Learnable parameters =", sum(p.numel() for p in self.cnn.parameters()))

        # Path to state dictionary
        self.state_dict_path = state_dict_path
        # Path and name for plots
        self.current = current
        self.params_name = params_name

        # Send net to GPU if possible
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pass

    def get_weights_transformed_for_sample(self, sample_weighting_method,
                                           no_of_classes, samples_per_cls,
                                           b_labels, beta=None):
        ''' Sample weighting '''

        def effective_num_of_samples(no_of_classes, beta, samples_per_cls):
            effective_num = 1.0 - np.power(beta, samples_per_cls)
            weights_for_samples = (1.0 - beta) / np.array(effective_num)
            weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
            return weights_for_samples

        def inverse_num_of_samples(no_of_classes, samples_per_cls, power=1):
            weights_for_samples = 1.0 / np.array(np.power(samples_per_cls,power))
            weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_of_classes
            return weights_for_samples

        if sample_weighting_method=='ens':
            weights_for_samples = effective_num_of_samples(no_of_classes, beta, samples_per_cls)
        elif sample_weighting_method=='ins':
            weights_for_samples = inverse_num_of_samples(no_of_classes, samples_per_cls)
        elif sample_weighting_method=='isns':
            weights_for_samples = inverse_num_of_samples(no_of_classes, samples_per_cls, power=0.5)
        else:
            return None
        b_labels = b_labels.to('cpu').numpy()
        weights_for_samples = torch.tensor(weights_for_samples).float()
        weights_for_samples = weights_for_samples.unsqueeze(0)
        weights_for_samples = torch.tensor(np.array(weights_for_samples.repeat(b_labels.shape[0],1) * b_labels))
        weights_for_samples = weights_for_samples.sum(1)
        weights_for_samples = weights_for_samples.unsqueeze(1)
        weights_for_samples = weights_for_samples.repeat(1, no_of_classes)

        return weights_for_samples

    def train_model(self, batch_size, learning_rate, num_epochs, amp, label,
                    spc=None, b=0.9, method='ens', validate=True, patience=10):
        # Initialize the early stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        # Define validation set and training set
        if validate:
            # Select 25% of data as validation
            ind_val = int(len(label) * 0.75)
            val_amp = amp[ind_val:]
            val_label = label[ind_val:]
            amp = amp[:ind_val]
            label = label[:ind_val]

        # Send model to device
        self.cnn.to(self.device)

        if spc==None:
            self.criterion = torch.nn.CrossEntropyLoss()  # for classification
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=learning_rate)

        # Try to load the model and optimizer 's state dictionaries
        self.load_model()

        # Train the model
        fig_loss = plt.figure(2)
        loss4plot = []
        val_loss4plot = []

        batches = []
        ind = 0
        while True:
            try:
                batches.append({'amp':torch.index_select(amp, 0, torch.tensor(np.int64(np.arange(ind,ind+batch_size,1)))),
                                'label':torch.index_select(label, 0, torch.tensor(np.int64(np.arange(ind,ind+batch_size,1)))),
                                'val_amp':torch.index_select(val_amp, 0, torch.tensor(np.int64(np.arange(ind,ind+batch_size,1)))),
                                'val_label':torch.index_select(val_label, 0, torch.tensor(np.int64(np.arange(ind,ind+batch_size,1))))})

                ind += batch_size
            except:
                break

        if (batches[-1]['amp']).size(0)!=batch_size:
            batches = batches[:-1]
            print("Removing last batch because of invalid batch size")

        for epoch in tqdm(range(num_epochs), total=num_epochs):
            hidden = None
            running_loss = 0.0
            val_running_loss = 0.0
            for batch in batches:
                if spc!=None:
                    # Sample weighting
                    sample_weighting_method = method
                    no_of_classes = 3
                    samples_per_cls = spc
                    b_labels = batch['label']
                    beta = b
                    weights = self.get_weights_transformed_for_sample(sample_weighting_method,
                                                                      no_of_classes,
                                                                      samples_per_cls,
                                                                      b_labels,
                                                                      beta=beta)

                    self.criterion = torch.nn.BCEWithLogitsLoss(weights.to(self.device))

                self.optimizer.zero_grad()

                outputs = self.cnn(batch['amp'].to(self.device))

                # Obtain the value for the loss function
                loss = self.criterion(outputs.to(self.device), batch['label'].to(self.device))

                running_loss += loss.item()

                loss.backward()

                self.optimizer.step()

                with torch.no_grad():
                    # Initialize model in testing mode
                    self.cnn.eval()
                    val_pred = self.cnn(batch['val_amp'].to(self.device))
                    val_loss = self.criterion(val_pred.to(self.device), batch['val_label'].to(self.device))

                    val_running_loss += val_loss.item()

                    # Initialize model in trainning mode again
                    self.cnn.train()

            # Early stop if validation loss increase
            early_stopping(val_running_loss/len(batches), self.cnn)

            if early_stopping.early_stop:
                print("-----------------")
                print("Early stopping...")
                print("-----------------")
                break

            loss4plot.append(running_loss/len(batches))
            val_loss4plot.append(val_running_loss/len(batches))

            # Plot loss vs epoch
            self.plot_loss(fig_loss, epoch, loss4plot, val_loss4plot)

            # Save model
            self.save_model(epoch, loss)
        pass

    def save_model(self, epoch, loss):
        # Save state dict of model in main folder
        torch.save({
                    #'epoch': epoch,
                    'model_state_dict': self.cnn.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    #'loss': loss,
                    }, self.state_dict_path)
        # Save state dict of model in current date folder
        torch.save({
                    #'epoch': epoch,
                    'model_state_dict': self.cnn.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    #'loss': loss,
                    }, self.current + '/' + self.state_dict_path)
        pass

    def load_model(self):
        # Load state dict of model
        try:
            checkpoint = torch.load(self.state_dict_path)
            self.cnn.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #epoch = checkpoint['epoch']
            #loss = checkpoint['loss']
            self.cnn.to(self.device)
            self.cnn.train()
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
