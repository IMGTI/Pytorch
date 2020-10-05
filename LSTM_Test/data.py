import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from main import current, params_name, seq_length, train_size

class Data(object):
    def __init__(self):
        pass

    def smooth(self, data, N_avg):
        def mov_avg(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)
        return mov_avg(data, N_avg)

    def ext_data(self, file, fig_name):
        data = pd.read_excel(file, fig_name, usecols=[0,1], names=['times', 'defs'])

        try:
            times = np.array([dt.datetime.timestamp(x) for x in data['times']])
        except:
            times = np.array(data['times'])

        # Convert times from seconds to days
        self.times = (times/(3600*24) -
                      (times/(3600*24))[0])

        self.defs = np.array(data['defs'])
        pass

    def data_smooth(self):
        # Apply Moving average

        self.N_avg = 2#5#2     # 2 para hacer una linea recta (tendencia) y al menos
                               # 5 puntos para tendencia valida (entonces con N_avg=2
                               # se logran 2-3 smooth ptos por cada 5)

        self.times = smooth(self.times, self.N_avg)
        self.defs = smooth(self.defs, self.N_avg)
        pass

    def select_lastwin(self):
        # Select last window of "seq_length" size
        self.lastt = self.times[-seq_length:]
        self.lastd = self.defs[-seq_length:]
        pass

    def reshape_data(self):
        # Reshape data array from 1D to 2D
        self.defs = defs.reshape(-1, 1)
        pass

    def sliding_windows(self, data, seq_length):
        x = []
        y = []

        for i in range(len(data)-seq_length-1):
            _x = data[i:(i+seq_length)]
            _y = data[i+seq_length]
            x.append(_x)
            y.append(_y)

        return np.array(x),np.array(y)

    def plot_data(self):
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

    def treat_data(self):
        # Load data into sequences
        training_set = self.defs

        sc = MinMaxScaler()
        training_data = sc.fit_transform(training_set)

        x, y = self.sliding_windows(training_data, seq_length)

        self.dataX = Variable(torch.Tensor(np.array(x)))
        self.dataY = Variable(torch.Tensor(np.array(y)))

        self.trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
        self.trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

        self.testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
        self.testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
        pass
