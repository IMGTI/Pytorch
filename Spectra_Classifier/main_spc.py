import os
import datetime as dt
from data_spc import Data
from train_spc import Train
from test_spc import Test
import getopt
import sys
import torch
import numpy as np

### Set RNG seeds

seed = 55

### Parse line arguments
def arg_parser(argv):
    train_arg = True
    test_arg = False
    num_epochs = 100
    test_file = ''
    data_path_arg = ''
    try:
        opts, args = getopt.getopt(argv,"hFt:r:e:n:",["train=","trdatapath=","tefile=","nepoch="])
    except getopt.GetoptError:
        print('argparser.py -t <True> -r <data_path> -e <test_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('------------ Commands -------------')
            print('-t: Train network (True/False)')
            print('-F: Force training and testing (-t True will not test)')
            print('-n: Number of epochs to use in training')
            print('-r: Path of files for training')
            print('-e: Input file for testing')
            print('-h: Show help')
            print('--------- Example usage ----------')
            print('For training:')
            print('argparser.py (-t True)/(-F) -r <data_path> -n <epochs>')
            print('For testing:')
            print('argparser.py -t False -e <test_file>')
            sys.exit()
        elif opt == '-F':
            train_arg = True
            test_arg = True
            print('Forcing training and testing...')
        elif opt in ("-r", "--trdatapath"):
            data_path_arg = arg
        elif opt in ("-e", "--tefile"):
            test_file = arg
        elif opt in ("-n", "--nepoch="):
            num_epochs = int(arg)
        elif opt in ("-t", "--train"):
            if arg=='True':
                train_arg = True
                test_arg = False
            else:
                train_arg = False
                test_arg = True
    if data_path_arg=='' and test_file=='':
        print('---------------------------------------------------')
        print('Please, enter a valid file for training or testing.')
        print('---------------------------------------------------')
    return (train_arg, test_arg, data_path_arg, test_file, num_epochs)

if __name__ == "__main__":
   train_arg, test_arg, data_path_arg, test_file, n_epochs = arg_parser(sys.argv[1:])

print('Train =', train_arg)
print('Test =', test_arg)
print('Train input file =', data_path_arg)
print('Test input file =', test_file)

### Classes

constituent_types = ['Albita',
                     'Alunita',
                     'Biotita',
                     'Ka_Pyr_Sm',
                     'Mus_Il_se',
                     'clor_cncl',
                     'se_gverde']

ind_constituent = 0

### Define the Hyperparameters

## Net parameters
constituent = constituent_types[ind_constituent]  # Select class model

num_epochs = n_epochs
learning_rate = [0.000307,0.000604,0.000478,0.001471,0.000191,0.001262,0.007655][ind_constituent]
input_size = 1
batch_size = [5,40,40,67,3,17,35][ind_constituent]
num_classes = 3
filters_number = [28,29,28,28,26,19,18][ind_constituent]
kernel_size = [3,3,5,1,2,1,4][ind_constituent]

## Data parameters
rd = True

## Train parameters
validate = True
patience = 20#10
sw = True
beta = 0.9998974
sample_method = 'ens'

## Parameters in name for .jpg files
params_name = ('_e' + str(num_epochs) +
               '_lr' + str(learning_rate) +
               '_b' + str(batch_size) +
               '_fn' + str(filters_number) +
               '_ks' + str(kernel_size) +
               '_i' + str(input_size) +
               '_o' + str(num_classes) +
               '_rd' + str(rd) +
               '_pat' + str(patience))

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
state_dict_path = 'state_dict_' + constituent

# Path to data

### Train
if train_arg:
    ## Extract data for training
    data = Data(seed)
    data_path = data_path_arg
    data.data_loader(data_path, constituent, current, random=rd)

    ## Train with data
    train = Train(batch_size,  input_size, num_classes, filters_number, kernel_size,
                  state_dict_path, current, params_name, seed)
    if sw:
        samples_per_cls = [data.yes, data.possible, data.no]

        train.train_model(batch_size, learning_rate, num_epochs, data.amp,
                          data.label, spc=samples_per_cls, b=beta, method=sample_method,
                          validate=validate, patience=patience)
    else:
        train.train_model(batch_size, learning_rate, num_epochs, data.amp,
                          data.label, validate=validate, patience=patience)

### Test
if test_arg:
    if test_file!='':
        # Extract data from input file
        data = Data(seed)
        data.test_data_loader(test_file, constituent, current)
        data.label = 'N/A'

    test = Test(batch_size, num_classes, input_size, filters_number, kernel_size,
                 state_dict_path, current, params_name, seed, tfile=test_file)

    test.test_model(data.amp, data.label, sc=data.scaler)

# Store parameters and runtime info in file
params_file = open(current + '/params.txt', 'w')

params_file.write('current  = ' + str(current) + '\n')
params_file.write('train_arg  = ' + str(train_arg) + '\n')
if train_arg:
    params_file.write('train_file_path  = ' + str(data_path) + '\n')
    params_file.write('num_epochs  = ' + str(num_epochs) + '\n')
else:
    params_file.write('train_file_path  = ' + '\n')
    params_file.write('num_epochs  = ' + '\n')
if sw:
    params_file.write('sample weighting  = ' + str(sw) +  '\n')
    params_file.write('beta  = ' + str(beta) +  '\n')
    params_file.write('sample method  = ' + sample_method +  '\n')
params_file.write('test_arg  = ' + str(test_arg) + '\n')
params_file.write('test_file  = ' + str(test_file) + '\n')
params_file.write('learning_rate  = ' + str(learning_rate) + '\n')
params_file.write('input_size  = ' + str(input_size) + '\n')
params_file.write('batch_size  = ' + str(batch_size) + '\n')
params_file.write('filters_number  = ' + str(filters_number) + '\n')
params_file.write('kernel_size  = ' + str(kernel_size) + '\n')
params_file.write('rd  = ' + str(rd) + '\n')
params_file.write('validate  = ' + str(validate) + '\n')
params_file.write('patience  = ' + str(patience) + '\n')
params_file.write('constituent  = ' + str(constituent) + '\n')

params_file.close()
