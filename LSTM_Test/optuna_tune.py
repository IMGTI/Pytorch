from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import optuna
from model import LSTM
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
import joblib
import datetime as dt
import pandas as pd
import getopt
import sys
import datetime as dt
from tqdm import tqdm

### Set RNG seeds

global seed
seed = 55

np.random.seed(seed)  # Numpy
torch.manual_seed(seed)  # Pytorch

### Parse line arguments
def arg_parser(argv):
    # Set device (Send to GPU if possible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set number of samples
    num_samples = 100
    try:
        opts, args = getopt.getopt(argv,"hn:d:c:",["nsamples=","device=", "chckpt="])
    except getopt.GetoptError:
        print('argparser.py -n <number_samples> -d <cpu/gpu> -c <chckpt>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('argparser.py -n <number_samples> -d <cpu/gpu> -c <chckpt>')
            sys.exit()
        elif opt in ("-n", "--nsamples"):
            num_samples = int(arg)
        elif opt in ("-d", "--device"):
            if arg=='cpu':
                device = torch.device(arg)
            elif arg=='gpu':
                device = torch.device("cuda:0")
        elif opt in ("-c", "--chckpt"):
            if arg==True:
                checkpoint = True
            else:
                checkpoint = False
    return num_samples, device, checkpoint

if __name__ == "__main__":
    num_samples, device, checkpoint = arg_parser(sys.argv[1:])
else:
    num_samples = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(na, data_dir):
    def ext_data(file):
        data = pd.read_excel(file, usecols=[0,1], names=['times', 'defs'])

        try:
            times = np.array([dt.datetime.timestamp(x) for x in data['times']])
        except:
            times = np.array(data['times'])

        # Convert times from seconds to days
        times = (times/(3600*24) -
                 (times/(3600*24))[0])

        defs = np.array(data['defs'])
        return times, defs

    def data_smooth(times, defs, N_avg=2):
        def mov_avg(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)

        # Apply Moving average

        #N_avg = 2#5#2     # 2 para hacer una linea recta (tendencia) y al menos
                          # 5 puntos para tendencia valida (entonces con N_avg=2
                          # se logran 2-3 smooth ptos por cada 5)

        times = mov_avg(times, N_avg)
        defs = mov_avg(defs, N_avg)
        return times, defs

    # Path to data
    data_path = 'datos'

    fig_num = 1
    file = data_path + '/Figura_de_control/Figura_de_control_desde_feb_fig' + str(fig_num) + '.xlsx'

    times, defs = ext_data(data_dir + '/' + file)
    times, defs = data_smooth(times, defs, N_avg=na)

    return times, defs

def treat_data(times, defs, seq_length, random_win=False):
    def random_win(x, y):
        ind_rand = np.random.permutation(len(y))
        rev_rand = np.argsort(ind_rand)
        return x[ind_rand], y[ind_rand], rev_rand
    def reshape_data(data):
        # Reshape data array from 1D to 2D
        data = data.reshape(-1, 1)
        return data

    def sliding_windows(data, seq_length):
        x = []
        y = []

        for i in range(len(data)-seq_length-1):
            _x = data[i:(i+seq_length)]
            _y = data[i+seq_length]
            x.append(_x)
            y.append(_y)

        return np.array(x),np.array(y)

    def scaling(data):
        # Reshape for scaling
        data = reshape_data(data)
        # Load scaler file if exists (if not, model is not trained)
        sc_filename = 'scaler_tune.save'
        try:
            scaler = joblib.load(sc_filename)
            data_sc = scaler.transform(data)
        except:
            print('Scaler save file not found. Probably due to not trained model. ')

            scaler = StandardScaler()
            data_sc = scaler.fit_transform(data)

            # Save scaler for later use in test
            joblib.dump(scaler, sc_filename)
        return data_sc

    # Load data into sequences
    training_set = defs

    training_data = scaling(training_set)

    # Treat data
    x, y = sliding_windows(training_data, seq_length)
    if random_win:
        x, y, rev_rand = random_win(x, y)

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x)))
    trainY = Variable(torch.Tensor(np.array(y)))

    # Times according with dataX and dataY dimensions
    time_step = np.absolute(times[0] - times[1])
    times_dataY = (times + (seq_length*time_step))[:-seq_length-1]

    return dataX, dataY, times_dataY, time_step, rev_rand

def train_model(trial):
    # Data directory
    data_dir = 'D:/Documents/GitHub/Pytorch/LSTM_Test'

    # Make validation while training
    validate = True

    # Model Parameters
    na = trial.suggest_int('na', 2, 2)
    do = trial.suggest_uniform('do', 0.01, 0.05)
    hs = trial.suggest_int('hs', 1, 100)#10)
    nl = trial.suggest_int('nl', 1, 10)#4)
    sl = trial.suggest_int('sl', 1, 200)#100)
    bs = trial.suggest_int('bs', 1, 100)#50)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    bd = trial.suggest_int('bd', 0, 1)
    st = trial.suggest_int('st', 0, 1)
    rd = trial.suggest_int('rd', 0, 1)

    # Training parameters
    max_nepochs = trial.suggest_int('max_nepochs', 10, 10)

    # Transform num into bool for biderectionality
    if bd==0:
        bid = False
    else:
        bid = True

    if st==1:
        stateful = True
    else:
        stateful = False

    if rd==1:
        rw = True
    else:
        rw = False
    # Load data
    times, defs = load_data(na, data_dir)
    defsX, defsY, times_dataY, time_step, rev_rand = treat_data(times, defs, sl, random_win=rw)

    # Initialize model
    lstm = LSTM(bs, 1, 1, hs, nl, do, bid, seed)
    # Send model to device
    lstm.to(device)

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

    # Train the model

    # Define validation set and training set
    if validate:
        ind_val = int(len(defsY) * 0.25)  # Select 25% of data as validation
        val_defsX = defsX[ind_val:]
        val_defsY = defsY[ind_val:]
        defsX = defsX[:ind_val]
        defsY = defsY[:ind_val]

    if bs==-1:
        for epoch in tqdm(range(max_nepochs), total=max_nepochs):
            optimizer.zero_grad()

            outputs, hidden = lstm(defsX.to(device))

            # Obtain the value for the loss function
            loss = criterion(outputs.to(device), defsY.to(device))

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                # Initialize model in testing mode
                lstm.eval()
                val_pred, val_hidden = lstm(val_defsX.to(device))
                val_loss = criterion(val_pred.to(device), val_defsY.to(device))

                loss4report = val_loss.item()
                # Initialize model in trainning mode again
                lstm.train()

            # Report loss to optuna
            trial.report(loss4report, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()
    else:
        batches = []
        ind = 0
        while True:
            try:
                batches.append({'defsX':torch.index_select(defsX, 0, torch.tensor(np.int64(np.arange(ind,ind+bs,1)))),
                                'defsY':torch.index_select(defsY, 0, torch.tensor(np.int64(np.arange(ind,ind+bs,1)))),
                                'val_defsX':torch.index_select(val_defsX, 0, torch.tensor(np.int64(np.arange(ind,ind+bs,1)))),
                                'val_defsY':torch.index_select(val_defsY, 0, torch.tensor(np.int64(np.arange(ind,ind+bs,1))))})

                ind += bs
            except:
                break

        if (batches[-1]['defsX']).size(0)!=bs:
            batches = batches[:-1]
            print("Removing last batch because of invalid batch size")


        for epoch in tqdm(range(max_nepochs), total=max_nepochs):
            hidden = None
            running_loss = 0.0
            val_running_loss = 0.0
            for batch in batches:
                optimizer.zero_grad()

                if stateful:
                    outputs, hidden = lstm(batch['defsX'].to(device), hidden=hidden)
                else:
                    outputs, hidden = lstm(batch['defsX'].to(device))

                # Obtain the value for the loss function
                loss = criterion(outputs.to(device), batch['defsY'].to(device))

                running_loss += loss.item()

                loss.backward()

                optimizer.step()

                with torch.no_grad():
                    # Initialize model in testing mode
                    lstm.eval()
                    val_pred, val_hidden = lstm(batch['val_defsX'].to(device))
                    val_loss = criterion(val_pred.to(device), batch['val_defsY'].to(device))

                    val_running_loss += val_loss.item()

                    # Initialize model in trainning mode again
                    lstm.train()

            loss4report = (val_running_loss/len(batches))

            # Report loss to optuna
            trial.report(loss4report, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

    return loss4report

def hyp_tune(num_samples=10, max_num_epochs=10):
    # Set sampler
    sampler = optuna.samplers.TPESampler()

    # Create optuna study
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), sampler=sampler,
                                direction='minimize')

    # Begin optimization
    study.optimize(train_model, n_trials=num_samples, n_jobs=1, gc_after_trial=True)

    # Dump into pickle file the results
    joblib.dump(study, 'optuna.pkl')
    #df_result = study.trials_dataframe()
    best_trial_params = study.best_params

    best_config = [study.best_value,
                   best_trial_params['hs'],
                   best_trial_params['nl'],
                   best_trial_params['sl'],
                   best_trial_params['bs'],
                   best_trial_params['lr'],
                   best_trial_params['na'],
                   best_trial_params['do'],
                   best_trial_params['bd'],
                   best_trial_params['st'],
                   best_trial_params['rd'],
                   best_trial_params['max_nepochs']]
    print('Best configuration parameters:')
    print('------------------------------')
    print(' Validation Loss = ', best_config[0], '\n',
          'Hidden Size = ', best_config[1], '\n',
          'Number of layers = ', best_config[2], '\n',
          'Sequence length = ', best_config[3], '\n',
          'Batch Size = ', best_config[4], '\n',
          'Learning rate = ', best_config[5], '\n',
          'Number for Moving Average = ', best_config[6], '\n',
          'Dropout = ', best_config[7], '\n',
          'Bidirectional (0:F 1:T) = ', best_config[8], '\n',
          'Stateful (0:F 1:T) = ', best_config[9], '\n',
          'Randomized Data (0:F 1:T) = ', best_config[10], '\n',
          'Maximum Number of Epochs Used = ', best_config[11])

    # Store best parameters in file
    best_params_file = open('best_params_optuna.txt', 'a')

    best_params_file.write('Date = ' + dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + '\n')
    best_params_file.write('Validation Loss = ' + str(best_config[0]) + '\n')
    best_params_file.write('Hidden Size = ' + str(best_config[1]) + '\n')
    best_params_file.write('Number of layers = ' + str(best_config[2]) + '\n')
    best_params_file.write('Sequence length = ' + str(best_config[3]) + '\n')
    best_params_file.write('Batch Size = ' + str(best_config[4]) + '\n')
    best_params_file.write('Learning rate = ' + str(best_config[5]) + '\n')
    best_params_file.write('Number for Moving Average = ' + str(best_config[6]) + '\n')
    best_params_file.write('Dropout = ' + str(best_config[7]) + '\n')
    best_params_file.write('Bidirectional (0:F 1:T) = ' + str(best_config[8]) + '\n')
    best_params_file.write('Stateful (0:F 1:T) = ' + str(best_config[9]) + '\n')
    best_params_file.write('Randomized Data (0:F 1:T) = ' + str(best_config[10]) + '\n')
    best_params_file.write('Number of Samples = ' + str(num_samples) + '\n')
    best_params_file.write('Maximum Number of Epochs Used = ' + str(best_config[11]) + '\n')
    best_params_file.write('\n')
    best_params_file.write('----------------------------------------------------' + '\n')
    best_params_file.write('\n')

    best_params_file.close()


if __name__ == "__main__":
    if device==torch.device("cuda:0"):
        hyp_tune(num_samples=num_samples, max_num_epochs=10)
    else:
        hyp_tune(num_samples=num_samples, max_num_epochs=10)
