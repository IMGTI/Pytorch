import numpy as np
import matplotlib.pyplot as plt
import torch
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1s
from model_spc import CNN
import pandas as pd
from pandas import ExcelWriter
from data_spc import Data

class Test(object):
    def __init__(self, batch_size, num_classes, input_size, filters_number, kernel_size,
                 state_dict_path, current, constituent, seed, tfile=''):
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
        self.constituent = constituent

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

    def save_results(self, whole_result, test_result):
        results_file = open('results.txt', 'a')
        results_file.write('-------------------------------------------------\n')
        results_file.write('Results folder = ' + str(self.current) + '\n')
        results_file.write('Constituent = ' + str(self.constituent) + '\n')
        results_file.write('Whole data results: ' + '\n')
        if len(whole_result)==1:
            results_file.write('Recall = ' + str(whole_result) + ' [%] \n')
        else:
            results_file.write('Precision (Median) = ' + str(whole_result[0]) + ' [%] \n')
            results_file.write('Recall = ' + str(whole_result[1]) + ' [%] \n')
            results_file.write('F1 Score (Micro Average) = ' + str(whole_result[2]) + '\n')
        results_file.write('Test data results: ' + '\n')
        if len(test_result)==1:
            results_file.write('Recall = ' + str(test_result) + ' [%] \n')
        else:
            results_file.write('Precision (Median) = ' + str(test_result[0]) + ' [%] \n')
            results_file.write('Recall = ' + str(test_result[1]) + ' [%] \n')
            results_file.write('F1 Score (Micro Average) = ' + str(test_result[2]) + '\n')
        results_file.close()

    def test_model(self, amp, label, amp_t, label_t):
        # List of classes
        classes = ['YES', 'POSSIBLE', 'NO']

        # Load model
        self.load_model()

        ### Training test
        try:
            # Performance in whole dataset

            correct = 0
            total = 0
            f1_labels = []
            f1_predicted = []
            precision = []
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

                    # Store for median precision
                    soft = torch.nn.Softmax(dim=1)
                    prec_outputs = soft(outputs)  # Obtain probabilities
                    prec_, prec_predicted = torch.max(prec_outputs.data, 1)
                    precision.append(np.array(prec_.cpu())[0])

                    # Store for F1 score
                    f1_labels.append(labels)
                    f1_predicted.append(predicted)

            print('Accuracy of the network on whole dataset (Recall): %d %%' % (
                100 * correct / total))

            whole_result = [100 * np.median(np.array(precision)),
                            100 * correct / total,
                            f1s(f1_labels, f1_predicted, average='micro')]

            # Performance in test dataset

            correct = 0
            total = 0
            f1_labels = []
            f1_predicted = []
            precision = []
            with torch.no_grad():
                for ind, input in enumerate(amp_t):
                    # Reshape input as shape of batch (batch_size,data_size,channels)
                    input = input.reshape(1, -1, 1)

                    outputs = self.cnn(input.to(self.device))
                    labels = np.where(np.array(label_t)[ind]==1)[0][0]
                    _, predicted = torch.max(outputs.data, 1)

                    predicted = np.array(predicted.cpu())[0]
                    if classes[predicted] == classes[labels]:
                        correct += 1

                    total += 1#labels.size(0)

                    # Store for median precision
                    soft = torch.nn.Softmax(dim=1)
                    prec_outputs = soft(outputs)  # Obtain probabilities
                    prec_, prec_predicted = torch.max(prec_outputs.data, 1)
                    precision.append(np.array(prec_.cpu())[0])

                    # Store for F1 score
                    f1_labels.append(labels)
                    f1_predicted.append(predicted)

            print('Accuracy of the network on test dataset (Recall): %d %%' % (
                100 * correct / total))

            test_result = [100 * np.median(np.array(precision)),
                           100 * correct / total,
                           f1s(f1_labels, f1_predicted, average='micro')]

            self.save_results(whole_result, test_result)

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
                    predicted_labels_prec = str(round(np.array(_.cpu())[0] * 100, 3))

                    print('Predicted: ' + predicted_labels)
                    print('Precision: ' + predicted_labels_prec + '%')
                    try:
                        print('Ground Truth: ' + ground_truth_labels)
                    except:
                        pass
