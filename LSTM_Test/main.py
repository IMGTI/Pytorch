import os
import datetime as dt
import torch
from data import Data
from train import Train
from test import Test
import getopt
import sys
import joblib

### Parse line arguments
def arg_parser(argv):
    train_arg = True
    test_arg = False
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hFt:i:",["train=","ifile="])
    except getopt.GetoptError:
        print('main.py -t <True> -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py (-t <[True]/False>) -i <inputfile>')
            sys.exit()
        elif opt == '-F':
            train_arg = True
            test_arg = True
            print('Forcing training and testing...')
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-t", "--train"):
            if arg=='True':
                train_arg = True
                test_arg = False
            else:
                train_arg = False
                test_arg = True
    return (train_arg, test_arg, inputfile)
if __name__ == "__main__":
   train_arg, test_arg, inputfile = arg_parser(sys.argv[1:])

print('Train =', train_arg)
print('Test =', test_arg)
print('Input file =', inputfile)

### Define the Hyperparameters

# Net parameters
num_epochs = 200#6000#200#300#2000
learning_rate = 0.001#0.001#0.01
input_size = 1
batch_size = 1  # Unused variable - Batch size is automatically handled (not 1)
hidden_size = 10#100#10#2
num_layers = 2#1
num_classes = 1

# Data parameters
seq_length = 12#1000#4  # Train Window
                        # 1h = 12
                        # 5min = 1
train_size = -100#int(len(y) * 0.67)
test_size = -100#len(y) - train_size  # Unused variable
fut_pred = 12#100  # Number of predictions
dropout = 0.05#0.05

# Parameters in name for .jpg files
params_name = ('_e' + str(num_epochs) +
               '_lr' + str(learning_rate) +
               '_b' + str(batch_size) +
               '_i' + str(input_size) +
               '_n' + str(num_layers) +
               '_h' + str(hidden_size) +
               '_o' + str(num_classes) +
               '_trw' + str(seq_length) +
               '_drp' + str(dropout))

### Create directory for each run and different hyperparameters

current = dt.datetime.now().strftime("%d_%m_%Y") + '/' + dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

# Create directory
try:
    os.mkdir(dt.datetime.now().strftime("%d_%m_%Y"))
    os.mkdir(current)
except:
    try:
        os.mkdir(current)
    except:
        pass

# Path for state dictionary to save model's weights and parameters
state_dict_path = 'state_dict'

### Train
if train_arg:
    ## Extract data for training

    #file = 'Figura de Control.xlsx'
    #file = 'prueba_serie.xlsx'
    fig_num = 1
    file = 'Figura_de_control_desde_feb_fig' + str(fig_num) + '.xlsx'

    data = Data()
    data.ext_data(file)
    data.data_smooth()
    data.plot_data(current, params_name)
    data.treat_data(train_size, seq_length)

    ## Train with data
    train = Train(num_classes, input_size, hidden_size, num_layers, dropout,
                  state_dict_path, current, params_name)
    train.train_model(learning_rate, num_epochs, data.times_dataY, data.dataX, data.dataY)

### Test
if test_arg:
    if inputfile!='':
        # Extract data from input file
        data = Data()
        data.ext_data(inputfile)
        #data.data_smooth()
        data.select_lastwin(seq_length)
        # Use last seq_length-data
        ind_test = -1
    else:
        # Use custom selected input from train data
        ind_test = -100#5000#1000#len(dataX)-1

    test = Test(num_classes, input_size, hidden_size, num_layers, dropout,
                state_dict_path, current, params_name)
    test.test_model(ind_test, seq_length, fut_pred, data.times_dataY, data.dataX,
                    data.dataY, sc=data.scaler)
