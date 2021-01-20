import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
import joblib
import os
import spc

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
        spc_data = spc.File(file).data_txt()  # Read data from file
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

    def data_smooth(self, N_avg=2):
        # Apply Moving average

        self.N_avg = N_avg#2#5#2     # 2 para hacer una linea recta (tendencia) y al menos
                               # 5 puntos para tendencia valida (entonces con N_avg=2
                               # se logran 2-3 smooth ptos por cada 5)

        self.times = self.smooth(self.times, self.N_avg)
        self.defs = self.smooth(self.defs, self.N_avg)
        pass

    def reshape_data(self, data):
        # Reshape data array from 1D to 2D
        data = data.reshape(-1, 1)
        return data

    def sliding_windows(self, data, seq_length):
        x = []
        y = []

        for i in range(len(data)-seq_length-1):
            _x = data[i:(i+seq_length)]
            _y = data[i+seq_length]
            x.append(_x)
            y.append(_y)

        return np.array(x),np.array(y)

    def sliding_windows_no_overlap(self, data, seq_length):
        size_overlap = 0.5

        x = []
        y = []

        try:
            i = 0
            while True:
                _x = data[i:(i+seq_length)]
                _y = data[i+seq_length]
                i = i+int(seq_length*size_overlap)
                x.append(_x)
                y.append(_y)
        except:
            return np.array(x),np.array(y)

    def random_win(self, x, y):
        ind_rand = np.random.permutation(len(y))
        self.rev_rand = np.argsort(ind_rand)
        return x[ind_rand], y[ind_rand]

    def plot_data(self, current, params_name):
        # Plot Data
        fig1 = plt.figure(1)
        fig1.clf()
        plt.plot(self.times, self.defs, 'r-', label = 'Raw Data')
        plt.title('Deformation vs Time')
        plt.ylabel('Defs(cm)')
        plt.xlabel('Time(d)')
        plt.grid(True)
        plt.legend()
        fig1.savefig(current + "/defs_vs_times" + params_name + ".jpg")
        pass

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

    def select_lastwin(self, seq_length, ind_test):
        # Select last window of "seq_length" size
        if ind_test==-1:
            self.times_dataY = self.times[-seq_length:]
            self.dataX = self.defs[-seq_length:]
            self.dataY = self.defs[-seq_length:]
            self.dataX, self.dataY = self.scaling(self.dataX), self.scaling(self.dataY)
        else:
            self.times_dataY = self.times[ind_test-seq_length+1:ind_test+1]
            # Since there is more data, use all available data for plot
            self.dataX = (self.defs[ind_test-seq_length+1:])[:2*seq_length]
            self.dataY = (self.defs[ind_test-seq_length+1:])[:2*seq_length]
            self.dataX, self.dataY = self.scaling(self.dataX)[:seq_length], self.scaling(self.dataY)
        pass


    def treat_data(self, train_size, seq_length, current):
        # Load data into sequences
        training_set = self.defs

        training_data = self.scaling(training_set, current=current)

        # Treat data
        x, y = self.sliding_windows(training_data, seq_length)

        self.dataX = np.array(x)
        self.dataY = np.array(y)

    def data_loader(self, data_path, n_avg, current, params_name, train_size, seq_length, random_win=False):
        from tqdm import tqdm
        for ind, file in enumerate(tqdm(os.listdir(data_path), total=len(os.listdir(data_path)))):
            self.ext_data(data_path + '/' + file)
            self.data_smooth(N_avg=n_avg)
            self.plot_data(current, params_name + '_dataset_' + str(ind))
            self.treat_data(train_size, seq_length, current)

            if ind==0:
                self.alldataX = self.dataX.copy()
                self.alldataY = self.dataY.copy()
            else:
                if len(self.alldataX)!=0 and len(self.dataX)!=0:
                    self.alldataX = np.vstack((self.alldataX, self.dataX))
                    self.alldataY = np.vstack((self.alldataY, self.dataY))
                elif len(self.dataX)!=0:
                    self.alldataX = self.dataX.copy()
                    self.alldataY = self.dataY.copy()

        # Randomized all windows
        if random_win:
            self.alldataX, self.alldataY = self.random_win(self.alldataX, self.alldataY)

        self.times_dataY = np.arange(len(self.alldataY))
        self.alldataX = Variable(torch.Tensor(np.array(self.alldataX)))
        self.alldataY = Variable(torch.Tensor(np.array(self.alldataY)))

        self.dataX = self.alldataX.detach().clone()
        self.dataY = self.alldataY.detach().clone()

        pass
