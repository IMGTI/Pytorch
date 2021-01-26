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

    def get_label(self, constituent, data_file_name, label_file_name):
        # Open file containing labels and other data
        label_file = pd.read_csv(label_file_name)

        # Match labels by sample
        ind_sample = np.where(label_file["SAMPLECODE"]==data_file_name[:-4])[0]
        labels_sample = label_file.iloc[ind_sample]

        # Match labels by constituent
        ind_constituent = np.where(labels_sample["MEASCONSTITUENT"]==constituent)
        labels_constituent = labels_sample.iloc[ind_constituent].values

        return labels_constituent

    def reshape_data(self, data):
        # Reshape data array from 1D to 2D
        data = data.reshape(-1, 1)
        return data

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
            data_sc = self.scaler.fit_transform(data)

            # Save scaler for later use in test and in current working directory
            joblib.dump(self.scaler, sc_filename)
            if current:
                joblib.dump(self.scaler, current + '/' + sc_filename)
        return data_sc

    def data_loader(self, files_list, labels_file, data_path, constituent, current, params_name):
        batch = {}
        for ind, file in enumerate(tqdm(files_list, total=len(files_list))):
            # Extract data and labels
            self.ext_data(data_path + '/' + file)
            self.amp = self.scaling(self.amp, current=current)
            self.label = self.get_label(constituent, file, labels_file)
            # Add to batch
            batch[file[:-4]] = {'amplitude':self.amp, 'label':self.label}

        return batch
