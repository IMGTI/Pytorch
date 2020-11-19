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
    num_epochs = 100
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hFt:i:n:",["train=","ifile=","nepoch="])
    except getopt.GetoptError:
        print('argparser.py -t <True> -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('------------ Commands -------------')
            print('-t: Train network (True/False)')
            print('-F: Force training and testing (-t True will not test)')
            print('-n: Number of epochs to use in training')
            print('-i: Input file for testing')
            print('-h: Show help')
            print('--------- Example usage ----------')
            print('For training:')
            print('argparser.py (-t True)/(-F) -n <epochs>')
            print('For testing:')
            print('argparser.py -t False -i <inputfile>')
            sys.exit()
        elif opt == '-F':
            train_arg = True
            test_arg = True
            print('Forcing training and testing...')
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-n", "--nepoch="):
            num_epochs = int(arg)
        elif opt in ("-t", "--train"):
            if arg=='True':
                train_arg = True
                test_arg = False
            else:
                train_arg = False
                test_arg = True
    return (train_arg, test_arg, inputfile, num_epochs)

if __name__ == "__main__":
   train_arg, test_arg, inputfile, n_epochs = arg_parser(sys.argv[1:])

print('Train =', train_arg)
print('Test =', test_arg)
print('Input file =', inputfile)

### Define the Hyperparameters

## Net parameters
num_epochs = n_epochs#10
learning_rate = 0.0076#0.0009539191097038075
input_size = 1
batch_size = 31#7   # Batch size is automatically handled in model
                    # if -1 then uses 1 batch of full data-length size
hidden_size = 6#9
num_layers = 1#2#1#3#1
num_classes = 1
bidirectional = False#True#False
dropout = 0#0.02898
# Stateful
stateful = False#True

## Data parameters
n_avg = 2#23#2
# Random windows for training
rw = True#False#True
#if rw:
#    stateful = False
#else:
#    stateful = True

## Test parameters
fut_pred = 84#21  # Number of predictions

## Train parameters
validate = True
seq_length = 84#9    # Train Window
                     # 1h = 12
                     # 5min = 1
train_size = -fut_pred  # Not necessarily equal to fut_pred

## Parameters in name for .jpg files
params_name = ('_e' + str(num_epochs) +
               '_lr' + str(learning_rate) +
               '_b' + str(batch_size) +
               '_i' + str(input_size) +
               '_n' + str(num_layers) +
               '_h' + str(hidden_size) +
               '_o' + str(num_classes) +
               '_trw' + str(seq_length) +
               '_bid' + str(bidirectional) +
               '_na' + str(n_avg) +
               '_rw' + str(rw) +
               '_drp' + str(dropout) +
               '_stf' + str(stateful))

### Create directory for each run and different hyperparameters

current_month = dt.datetime.now().strftime("%m_%Y")
current_day = dt.datetime.now().strftime("%d_%m_%Y")
current_min = dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

current = (current_month + '/' +
           current_day + '/' +
           current_min)

# Create directory
try:
    os.mkdir(current_month)
    os.mkdir(current_month + '/' +
             current_day)
    os.mkdir(current)
except:
    try:
        os.mkdir(current_month + '/' +
                 current_day)
        os.mkdir(current)
    except:
        try:
            os.mkdir(current)
        except:
            pass

# Path for state dictionary to save model's weights and parameters
state_dict_path = 'state_dict'

# Path to data
data_path = 'datos'

### Train
if train_arg:
    ## Extract data for training

    #file = 'Figura de Control.xlsx'
    #file = 'prueba_serie.xlsx'
    fig_num = 2
    file = data_path + '/Figura_de_control/Figura_de_control_desde_feb_fig' + str(fig_num) + '.xlsx'
    #file = data_path + '/datos_26102020/datos_26102020_fig_' + str(fig_num) + '.xlsx'
    #file = 'prueba_serie.xlsx'

    data = Data()
    data.ext_data(file)
    data.data_smooth(N_avg=n_avg)
    data.plot_data(current, params_name)
    data.treat_data(train_size, seq_length, random_win=rw)

    ## Train with data
    train = Train(batch_size, num_classes, input_size, hidden_size, num_layers, dropout,
                  bidirectional, state_dict_path, current, params_name, stateful=stateful)
    train.train_model(batch_size, learning_rate, num_epochs, data.times_dataY,
                      data.dataX, data.dataY, validate=validate)

### Test
if test_arg:
    if inputfile!='':
        # Extract data from input file
        data = Data()
        data.ext_data(inputfile)
        data.data_smooth(N_avg=n_avg)

        # Use last seq_length-data
        ind_test = -1002#-fut_pred-2#-seq_length#-1
        data.select_lastwin(seq_length, ind_test)
    else:
        # Use custom selected input from train data
        ind_test = -1000#-fut_pred#5000#1000#len(dataX)-1

    test = Test(batch_size, num_classes, input_size, hidden_size, num_layers, dropout,
                bidirectional, state_dict_path, current, params_name)

    # Reorder data to original state just for test and train forcing
    if train_arg and rw:
        test.include_rw(data.rev_rand)

    test.test_model(ind_test, seq_length, fut_pred, data.times_dataY, data.dataX,
                    data.dataY, sc=data.scaler)

# Store parameters and runtime info in file
params_file = open(current + '/params.txt', 'w')

params_file.write('current  = ' + str(current) + '\n')
params_file.write('train_arg  = ' + str(train_arg) + '\n')
params_file.write('test_arg  = ' + str(test_arg) + '\n')
params_file.write('inputfile  = ' + str(inputfile) + '\n')
params_file.write('ind_test  = ' + str(ind_test) + '\n')
if train_arg:
    params_file.write('file  = ' + str(file) + '\n')
    params_file.write('num_epochs  = ' + str(num_epochs) + '\n')
else:
    params_file.write('file  = ' + '\n')
    params_file.write('num_epochs  = ' + '\n')
params_file.write('learning_rate  = ' + str(learning_rate) + '\n')
params_file.write('input_size  = ' + str(input_size) + '\n')
params_file.write('batch_size  = ' + str(batch_size) + '\n')
params_file.write('hidden_size  = ' + str(hidden_size) + '\n')
params_file.write('num_layers  = ' + str(num_layers) + '\n')
params_file.write('num_classes  = ' + str(num_classes) + '\n')
params_file.write('bidirectional  = ' + str(bidirectional) + '\n')
params_file.write('dropout  = ' + str(dropout) + '\n')
params_file.write('stateful  = ' + str(stateful) + '\n')
params_file.write('n_avg  = ' + str(n_avg) + '\n')
params_file.write('rw  = ' + str(rw) + '\n')
params_file.write('fut_pred  = ' + str(fut_pred) + '\n')
params_file.write('validate  = ' + str(validate) + '\n')
params_file.write('seq_length  = ' + str(seq_length) + '\n')
params_file.write('train_size  = ' + str(train_size) + '\n')

params_file.close()
