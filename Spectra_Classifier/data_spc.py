import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
import joblib
import os
import spc
import sys
from scipy import stats
from tqdm import tqdm

class Data(object):
    def __init__(self, seed):
        # RNG Seed
        np.random.seed(seed)  # Numpy
        torch.manual_seed(seed)  # Pytorch
        pass

    def ext_data(self, file):
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

        self.amp, ind_beg_end = fill_data(np.array(amp))

        # Erase NaNs values at the beginning and at the end
        if ind_beg_end:
            self.amp = self.amp[ind_beg_end[0]:ind_beg_end[1]]
            self.wave = self.wave[ind_beg_end[0]:ind_beg_end[1]]

        pass

    def random_shuffle(self, x, y):
        ind_rand = np.random.permutation(len(y))
        self.rev_rand = np.argsort(ind_rand)
        return x[ind_rand], y[ind_rand]

    def get_label(self, constituent, data_path, data_file_name, label_file_name):
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

    def reshape_data(self, data):
        # Reshape data array from 1D to 2D
        data = data.reshape(-1, 1)
        return data

    def reorder_windows(self, data, seq_length):
        x = []

        _x = data
        x.append(_x)

        return np.array(x)

    def treat_data(self, current):
        # Scale data
        self.amp = self.scaling(self.amp, current=current)
        # Reshape data into "windows" (just a reshape for working model)
        self.amp = self.reorder_windows(self.amp, len(self.amp))

        return self.amp

    def scaling(self, data, current=None):
        # Reshape for scaling
        data = self.reshape_data(data)
        # Load scaler file if exists (if not, model is not trained)
        sc_filename = 'scaler.save'
        try:
            self.scaler = joblib.load(sc_filename)
            data_sc = self.scaler.transform(data)
            # Save scaler in current working directory
            if current:
                joblib.dump(self.scaler, current + '/' + sc_filename)
        except:
            print('Scaler save file not found. Probably due to not trained model. ')

            self.scaler = StandardScaler()
            data_sc = self.scaler.fit_transform(data)  # numpy array

            # Save scaler for later use in test and in current working directory
            joblib.dump(self.scaler, sc_filename)
            if current:
                joblib.dump(self.scaler, current + '/' + sc_filename)

        return data_sc

    def test_data_loader(self, test_file, constituent, current):
        # Extract data and labels
        self.ext_data(test_file)  # Include path
        self.amp = self.treat_data(current=current)
        self.label = 'N/A'

        self.amp = Variable(torch.Tensor(np.array(self.amp)))

        pass

    def data_loader(self, data_path, constituent, current, random=False):
        data_file = 'data.ts'

        # Load previous loaded data if possible
        if data_file in os.listdir():
            print('Using previous loaded data...')
            self.amp, self.label, [self.yes, self.possible, self.no] = joblib.load(data_file)

        else:
            print('No previous data found. Loading data...')
            files_list = os.listdir(data_path)
            labels_file = np.array(files_list)[['.csv' in x for x in files_list]][0]
            files_list.remove(labels_file)

            self.yes = 0
            self.possible = 0
            self.no = 0

            for ind, file in enumerate(tqdm(files_list, total=len(files_list))):
                # Extract data and labels
                self.ext_data(data_path + '/' + file)
                self.amp = self.treat_data(current=current)

                # Skip data without proper label
                try:
                    self.label = self.get_label(constituent, data_path, file, labels_file)
                except:
                    continue
                # Add to data
                if ind==0:
                    self.all_amp = self.amp.copy()
                    self.all_label = self.label.copy()
                else:
                    if len(self.all_amp)!=0 and len(self.amp)!=0:
                        self.all_amp = np.vstack((self.all_amp, self.amp))
                        self.all_label = np.vstack((self.all_label, self.label))
                    elif len(self.amp)!=0:
                        self.all_amp = self.amp.copy()
                        self.all_label = self.label.copy()

                # Store number of samples per class
                if (self.label == np.array([1,0,0])).all():
                    self.yes +=1
                elif (self.label == np.array([0,1,0])).all():
                    self.possible +=1
                elif (self.label == np.array([0,0,1])).all():
                    self.no +=1

            # Randomized all windows
            if random:
                self.all_amp, self.all_label = self.random_shuffle(self.all_amp, self.all_label)

            self.all_amp = Variable(torch.Tensor(np.array(self.all_amp)))
            self.all_label = Variable(torch.Tensor(np.array(self.all_label)))

            self.amp = self.all_amp.detach().clone()
            self.label = self.all_label.detach().clone()

            # Save data for speeding up next execution
            joblib.dump((self.amp, self.label, [self.yes, self.possible, self.no]), data_file)

        pass
