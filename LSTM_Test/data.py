import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
import joblib

class Data(object):
    def __init__(self, seed):
        # RNG Seed
        np.random.seed(seed)  # Numpy
        torch.manual_seed(seed)  # Pytorch
        pass

    def smooth(self, data, N_avg):
        def mov_avg(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)
        return mov_avg(data, N_avg)

    def ext_data(self, file):
        def fill_data(y):
            # Look for nans
            ind_fill = np.where(np.isnan(y))[0]

            if len(ind_fill)==0:
                return y
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

            return y

        data = pd.read_excel(file, usecols=[0,1], names=['times', 'defs'])

        try:
            times = np.array([dt.datetime.timestamp(x) for x in data['times']])
        except:
            times = np.array(data['times'])

        # Convert times from seconds to days
        self.times = (times/(3600*24) -
                      (times/(3600*24))[0])

        self.defs = fill_data(np.array(data['defs']))
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


    def treat_data(self, train_size, seq_length, current, random_win=False):
        # Load data into sequences
        training_set = self.defs

        training_data = self.scaling(training_set, current=current)

        # Treat data
        x, y = self.sliding_windows(training_data, seq_length)
        if random_win:
            x, y = self.random_win(x, y)

        self.dataX = Variable(torch.Tensor(np.array(x)))
        self.dataY = Variable(torch.Tensor(np.array(y)))

        self.trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
        self.trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

        self.testX = Variable(torch.Tensor(np.array(x[train_size:])))
        self.testY = Variable(torch.Tensor(np.array(y[train_size:])))

        # Times according with dataX and dataY dimensions
        time_step = np.absolute(self.times[0] - self.times[1])
        self.times_dataY = (self.times + (seq_length*time_step))[:-seq_length-1]
