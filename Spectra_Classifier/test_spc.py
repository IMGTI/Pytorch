import numpy as np
import matplotlib.pyplot as plt
import torch
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as r2s
from model_spc import CNN
import pandas as pd
from pandas import ExcelWriter
from data_spc import Data

class Test(object):
    def __init__(self, batch_size, num_classes, input_size, filters_number, kernel_size,
                 state_dict_path, current, params_name, seed, tfile=''):
        # RNG Seed
        np.random.seed(seed)  # Numpy
        torch.manual_seed(seed)  # Pytorch

        # Save test file
        if tfile!='':
            self.test_file = tfile

        # Initialize the model
        self.cnn = CNN(input_size, num_classes, filters_number, kernel_size, seed)
        # Path to state dictionary
        self.state_dict_path = state_dict_path
        # Path and name for plots
        self.current = current
        self.params_name = params_name

        # Send net to GPU if possible
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pass

    def load_model(self):
        # Load state dict of model
        try:
            checkpoint = torch.load(self.state_dict_path)
            self.cnn.load_state_dict(checkpoint['model_state_dict'])
            self.cnn.to(self.device)
            self.cnn.eval()
        except:
            print('State dict(s) missing')
        pass

    def include_rd(self, ind):
        self.rev_rand = ind
        pass

    def test_model(self, amp, label, sc=None):
        # List of classes
        classes = ['YES', 'POSSIBLE', 'NO']

        # Load model
        self.load_model()

        ### Training test
        try:
            # Performance in whole dataset

            correct = 0
            total = 0
            with torch.no_grad():
                for ind, input in enumerate(amp):
                    # Reshape input as shape of batch (batch_size,data_size,channels)
                    input = input.reshape(1, -1, 1)

                    outputs = self.cnn(input.to(self.device))
                    labels = np.where(np.array(label)[ind]==1)[0][0]
                    _, predicted = torch.max(outputs.data, 1)

                    predicted = np.array(predicted.cpu())[0]
                    if classes[predicted] == classes[labels]:
                        correct += 1

                    total += 1#labels.size(0)

            print('Accuracy of the network on whole dataset: %d %%' % (
                100 * correct / total))

        ### Data Test (Classification)
        except:
            # Classifier
            with torch.no_grad():
                for input in amp:
                    outputs = self.cnn(amp.to(self.device))
                    _, predicted = torch.max(outputs.data, 1)
                    try:
                        ground_truth_labels = classes[label]
                    except:
                        pass
                    predicted_labels = classes[predicted]

                    print('Predicted: ' + predicted_labels)
                    try:
                        print('Ground Truth: ' + ground_truth_labels)
                    except:
                        pass
