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
from model_spc import CNN
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import getopt
import sys
from tqdm import tqdm
import spc
import datetime as dt

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


def ext_data(file):
    def fill_data(y):
        # Look for nans
        ind_fill = np.where(np.isnan(y))[0]

        if len(ind_fill)==0:
            return y, None
        else:
            # Form groups of ids
            ind_fill_groups = []

            i = 0
            for ind, val in enumerate(ind_fill):
                if ind!=0:
                    if np.absolute(ind_fill[ind-1]-ind_fill[ind])>1:
                        ind_inicial = i
                        ind_final = ind
                        # Add index group
                        ind_fill_groups.append(ind_fill[ind_inicial:ind_final])
                        # Set inicial index to new one
                        i = ind_final
            if ind_fill_groups==[]:
                ind_fill_groups.append(ind_fill)

            for ind_fill in ind_fill_groups:
                # Select boundary values
                try:
                    boundaries = (y[ind_fill[0]-1], y[ind_fill[-1]+1])
                except:
                    try:
                        boundaries = (y[ind_fill[-1]+1], y[ind_fill[-1]+1])
                    except:
                        boundaries = (y[ind_fill[0]-1], y[ind_fill[0]-1])

                # Fill with random data between boundaries
                for ind in ind_fill:
                    fill = np.random.uniform(boundaries[0], boundaries[1])
                    y[ind] = fill

            # Return indices of NaNs at the beginning and at the end
            ind_beg = 0
            ind_end = len(y)-1
            if ind_fill_groups[0][0]==0:
                ind_beg = ind_fill_groups[0][-1]  # Last index of first group
            if ind_fill_groups[-1][-1]==(len(y)-1):
                ind_end = ind_fill_groups[-1][0]  # First index of last group

            return y, (ind_beg, ind_end)

    # Extract data from .spc file

    # Block printing
    sys.stdout = open(os.devnull, 'w')

    spc_data = spc.File(file).data_txt()  # Read data from file

    # Enable printing
    sys.stdout = sys.__stdout__

    a = spc_data.split('\n')  # Format data
    b = [x.split('\t') for x in a]
    b.remove([''])
    wave, amp = np.array([[float(x[0]),float(x[1])] for x in b]).transpose()

    amp, ind_beg_end = fill_data(np.array(amp))

    # Erase NaNs values at the beginning and at the end
    if ind_beg_end:
        amp = amp[ind_beg_end[0]:ind_beg_end[1]]
        wave = wave[ind_beg_end[0]:ind_beg_end[1]]

    return wave, amp

def random_shuffle(x, y):
    ind_rand = np.random.permutation(len(y))
    rev_rand = np.argsort(ind_rand)
    return x[ind_rand], y[ind_rand]

def get_label(constituent, data_path, data_file_name, label_file_name):
    # Open file containing labels and other data
    label_file = pd.read_csv(data_path + "/" + label_file_name)

    # Match labels by sample
    ind_sample = np.where(label_file["SAMPLECODE"]==data_file_name[:-4])[0]
    labels_sample = label_file.iloc[ind_sample]

    # Match labels by constituent
    ind_constituent = np.where(labels_sample["MEASCONSTITUENT"]==constituent)[0]
    labels_constituent = labels_sample.iloc[ind_constituent]["MEASMATCH"].values[0]  # numpy array
    # One-hot encoding
    if labels_constituent=='YES' or labels_constituent=='yes' or labels_constituent=='Yes':
        labels_constituent = np.array([1,0,0])
    elif labels_constituent=='POSSIBLE' or labels_constituent=='possible' or labels_constituent=='Possible':
        labels_constituent = np.array([0,1,0])
    elif labels_constituent=='NO' or labels_constituent=='no' or labels_constituent=='No':
        labels_constituent = np.array([0,0,1])

    return labels_constituent

def reshape_data(data):
    # Reshape data array from 1D to 2D
    data = data.reshape(-1, 1)
    return data

def reorder_windows(data, seq_length):
    x = []

    _x = data
    x.append(_x)

    return np.array(x)

def treat_data(amp):
    # Scale data
    amp = scaling(amp)
    # Reshape data into "windows" (just a reshape for working model)
    amp = reorder_windows(amp, len(amp))

    return amp

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
        data_sc = scaler.fit_transform(data)  # numpy array

        # Save scaler for later use in test and in current working directory
        joblib.dump(scaler, sc_filename)

    return data_sc

def data_loader(data_path, constituent, random=False):
    data_file = 'data.ts'

    # Load previous loaded data if possible
    if data_file in os.listdir():
        print('Using previous loaded data...')
        amp, label = joblib.load(data_file)

    else:
        print('No previous data found. Loading data...')
        files_list = os.listdir(data_path)
        labels_file = np.array(files_list)[['.csv' in x for x in files_list]][0]
        files_list.remove(labels_file)
        '''
        yes = 0
        possible = 0
        no = 0
        '''
        for ind, file in enumerate(tqdm(files_list, total=len(files_list))):
            # Extract data and labels
            wave, amp = ext_data(data_path + '/' + file)
            amp = treat_data(amp)

            # Skip data without proper label
            try:
                label = get_label(constituent, data_path, file, labels_file)
            except:
                continue
            # Add to data
            if ind==0:
                all_amp = amp.copy()
                all_label = label.copy()
            else:
                if len(all_amp)!=0 and len(amp)!=0:
                    all_amp = np.vstack((all_amp, amp))
                    all_label = np.vstack((all_label, label))
                elif len(amp)!=0:
                    all_amp = amp.copy()
                    all_label = label.copy()
            '''
            # Store number of samples per class
            if (label == np.array([1,0,0])).all():
                yes +=1
            elif (label == np.array([0,1,0])).all():
                possible +=1
            elif (label == np.array([0,0,1])).all():
                no +=1
            '''
        # Randomized all windows
        if random:
            all_amp, all_label = random_shuffle(all_amp, all_label)

        all_amp = Variable(torch.Tensor(np.array(all_amp)))
        all_label = Variable(torch.Tensor(np.array(all_label)))

        amp = all_amp.detach().clone()
        label = all_label.detach().clone()

        # Save data for speeding up next execution
        joblib.dump((amp, label), data_file)

    return amp, label


def hyp_tune(constituent, num_samples=10, max_num_epochs=10):
    # Data directory
    data_path = 'D:/Documents/GitHub/Datos_Espectros/Espectros_analizados/PredMeasure'
    '''
    # Constituent
    constituent_types = ['Albita',
                         'Alunita',
                         'Biotita',
                         'Ka_Pyr_Sm',
                         'Mus_Il_se',
                         'clor_cncl',
                         'se_gverde']

    ind_constituent = 0

    constituent = constituent_types[ind_constituent]
    '''
    # Load data
    amp, label = data_loader(data_path, constituent, random=True)

    def train_model(trial, amp=amp, label=label):
        # Make validation while training
        validate = True

        # Model Parameters
        bs = trial.suggest_int('bs', 1, 100)
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
        fil = trial.suggest_int('fn', 1, 30)
        ker = trial.suggest_int('ks', 1, 5)

        # Training parameters
        max_nepochs = trial.suggest_int('max_nepochs', 10, 10)

        # Initialize model
        cnn = CNN(1, 3, fil, ker, seed)

        # Send model to device
        cnn.to(device)

        criterion = torch.nn.MSELoss()    # mean-squared error for regression
        optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

        # Train the model

        # Define validation set and training set
        if validate:
            # Select 25% of data as validation
            ind_val = int(len(label) * 0.75)
            val_amp = amp[ind_val:]
            val_label = label[ind_val:]
            amp = amp[:ind_val]
            label = label[:ind_val]

        # Send model to device
        cnn.to(device)

        criterion = torch.nn.MSELoss()    # mean-squared error for regression
        optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

        # Train the model
        batches = []
        ind = 0
        while True:
            try:
                batches.append({'amp':torch.index_select(amp, 0, torch.tensor(np.int64(np.arange(ind,ind+bs,1)))),
                                'label':torch.index_select(label, 0, torch.tensor(np.int64(np.arange(ind,ind+bs,1)))),
                                'val_amp':torch.index_select(val_amp, 0, torch.tensor(np.int64(np.arange(ind,ind+bs,1)))),
                                'val_label':torch.index_select(val_label, 0, torch.tensor(np.int64(np.arange(ind,ind+bs,1))))})

                ind += bs
            except:
                break

        if (batches[-1]['amp']).size(0)!=bs:
            batches = batches[:-1]
            print("Removing last batch because of invalid batch size")

        for epoch in tqdm(range(max_nepochs), total=max_nepochs):
            hidden = None
            running_loss = 0.0
            val_running_loss = 0.0
            for batch in batches:
                '''
                # Sample weighting
                sample_weighting_method = 'ens'
                no_of_classes = 3
                samples_per_cls = samp_per_cls
                b_labels = batch['label']
                beta = 0.9
                weights = get_weights_transformed_for_sample(sample_weighting_method,
                                                                  no_of_classes,
                                                                  samples_per_cls,
                                                                  b_labels,
                                                                  beta=beta)

                criterion = torch.nn.BCEWithLogitsLoss(weights.to(device))
                '''
                optimizer.zero_grad()

                outputs = cnn(batch['amp'].to(device))

                # Obtain the value for the loss function
                loss = criterion(outputs.to(device), batch['label'].to(device))

                running_loss += loss.item()

                loss.backward()

                optimizer.step()

                with torch.no_grad():
                    # Initialize model in testing mode
                    cnn.eval()
                    val_pred = cnn(batch['val_amp'].to(device))
                    val_loss = criterion(val_pred.to(device), batch['val_label'].to(device))

                    val_running_loss += val_loss.item()

                    # Initialize model in trainning mode again
                    cnn.train()

            loss4report = (val_running_loss/len(batches))

            # Report loss to optuna
            trial.report(loss4report, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

        return loss4report


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
                   best_trial_params['bs'],
                   best_trial_params['lr'],
                   best_trial_params['ks'],
                   best_trial_params['fn'],
                   best_trial_params['max_nepochs']]
    print('Best configuration parameters:')
    print('------------------------------')
    print(' Validation Loss = ', best_config[0], '\n',
          'Batch Size = ', best_config[1], '\n',
          'Learning rate = ', best_config[2], '\n',
          'Kernel Size = ', best_config[3], '\n',
          'Number of Filters = ', best_config[4], '\n',
          'Maximum Number of Epochs Used = ', best_config[5])

    # Store best parameters in file
    best_params_file = open('best_params_optuna_' + constituent + '.txt', 'a')

    best_params_file.write('Date = ' + dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + '\n')
    best_params_file.write('Validation Loss = ' + str(best_config[0]) + '\n')
    best_params_file.write('Batch Size = ' + str(best_config[1]) + '\n')
    best_params_file.write('Learning rate = ' + str(best_config[2]) + '\n')
    best_params_file.write('Kernel Size = ' + str(best_config[3]) + '\n')
    best_params_file.write('Number of Filters = ' + str(best_config[4]) + '\n')
    best_params_file.write('Number of Samples = ' + str(num_samples) + '\n')
    best_params_file.write('Maximum Number of Epochs Used = ' + str(best_config[5]) + '\n')
    best_params_file.write('\n')
    best_params_file.write('----------------------------------------------------' + '\n')
    best_params_file.write('\n')

    best_params_file.close()

'''
if __name__ == "__main__":
    hyp_tune(num_samples=num_samples, max_num_epochs=10)
'''
# Constituent
constituent_types = ['Albita',
                     'Alunita',
                     'Biotita',
                     'Ka_Pyr_Sm',
                     'Mus_Il_se',
                     'clor_cncl',
                     'se_gverde']

if __name__ == "__main__":
    for constituent in constituent_types:
        hyp_tune(constituent, num_samples=num_samples, max_num_epochs=10)
