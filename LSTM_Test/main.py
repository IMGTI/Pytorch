import os
import datetime as dt
import torch
from data import Data
from train import Train
from test import Test
import getopt
import sys

### Parse line arguments
def arg_parser(argv):
    train_arg = True
    test_arg = False
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hFt:i:",["train=","ifile="])
    except getopt.GetoptError:
        print('argparser.py -t <True> -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('argparser.py (-t <[True]/False>) -i <inputfile>')
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
num_epochs = 2#10#100#200#1000#300#2000
learning_rate = 0.0008695868177968809#0.0003910427505590165#0.022472643513504736#0.001#0.001#0.01
input_size = 1
batch_size = 27#50  # Batch size is automatically handled in model
                    # if -1 then uses 1 batch of full data-length size
hidden_size = 8#5#10#100#10#2
num_layers = 2#1#3#1
num_classes = 1
bidirectional = False#True
dropout = 0.031194832470140016#0.05#0#0.05
fut_pred = 21#92#200#12#100  # Number of predictions

# Data parameters
seq_length = 21#72#92#12#1000#4  # Train Window
                        # 1h = 12
                        # 5min = 1
train_size = -100#int(len(y) * 0.67)
test_size = -100#len(y) - train_size  # Unused variable

n_avg = 2#43#2

# Random windows for training
rw = False#True

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
    #file = 'prueba_serie.xlsx'

    data = Data()
    data.ext_data(file)
    data.data_smooth(N_avg=n_avg)
    data.plot_data(current, params_name)
    data.treat_data(train_size, seq_length, random_win=rw)

    ## Train with data
    train = Train(num_classes, input_size, hidden_size, num_layers, dropout,
                  bidirectional, state_dict_path, current, params_name)
    train.train_model(batch_size, learning_rate, num_epochs, data.times_dataY,
                      data.dataX, data.dataY)

### Test
if test_arg:
    if inputfile!='':
        # Extract data from input file
        data = Data()
        data.ext_data(inputfile)
        data.data_smooth(N_avg=n_avg)

        # Use last seq_length-data
        ind_test = -1002#-seq_length#-1
        data.select_lastwin(seq_length, ind_test)
    else:
        # Use custom selected input from train data
        ind_test = -1000#5000#1000#len(dataX)-1

    test = Test(num_classes, input_size, hidden_size, num_layers, dropout,
                bidirectional, state_dict_path, current, params_name)

    # Reorder data to original state just for test and train forcing
    if train_arg and rw:
        test.include_rw(data.rev_rand)

    test.test_model(ind_test, seq_length, fut_pred, data.times_dataY, data.dataX,
                    data.dataY, sc=data.scaler)
