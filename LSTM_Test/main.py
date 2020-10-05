import os
import datetime as dt
import torch
import sys
import getopt
from data import Data
from train import Train
from test import Test

### Parse line arguments

def main(argv):
    train_arg = True
    test_arg = False
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"ht:i:o:",["train=","ifile=","ofile="])
    except getopt.GetoptError:
        print 'main.py -t <True> -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'main.py (-t <[True]/False>) -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-t", "--train"):
            train_arg = arg
            if arg==True:
                test_arg = False
            else:
                test_arg = True
if __name__ == "__main__":
   main(sys.argv[1:])


### Define the Hyperparameters

# Net parameters
num_epochs = 2000#200#300#2000
learning_rate = 0.001#0.001#0.01
input_size = 1
batch_size = 1  # Unused variable
hidden_size = 100#10#2
num_layers = 1

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

### Select Device

# Send net to GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Data

#file = 'Figura de Control.xlsx'
#fig_name = 'F6'
#file = 'prueba_serie.xlsx'
#fig_name = 'Sheet1'
file = 'Figura_de_control_desde_feb.xlsx'
fig_name = 'Datos'

data = Data()
data.ext_data(file, fig_name)
data.data_smooth()
data.reshape_data()
data.plot_data()
data.treat_data()

### Train
if train_arg:
    train = Train()



### Test
inputfile = inputfile
outputfile = outputfile
if test_arg:
    test = Test()
