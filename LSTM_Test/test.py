import numpy as np
import matplotlib.pyplot as plt
import torch
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score as r2s
from model import LSTM

class Test(object):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout,
                 state_dict_path, current, params_name):
        # Initialize the model
        self.lstm = LSTM(num_classes, input_size, hidden_size, num_layers, dropout)
        # Path to state dictionary
        self.state_dict_path = state_dict_path
        # Path and name for plots
        self.current = current
        self.params_name = params_name

        # Send net to GPU if possible
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pass

    def load_model(self):
        # Load state dict of model
        try:
            checkpoint = torch.load(self.state_dict_path)
            self.lstm.load_state_dict(checkpoint['model_state_dict'])
            self.lstm.to(self.device)
            self.lstm.eval()
        except:
            print('State dict(s) missing')
        pass

    def plot_predict(self, fig_name, x, y, seq_length, fut_pred):
        fig_pred = plt.figure(3)
        fig_pred.clf()
        plt.plot(x[0], y[0], 'r-', label = 'Raw Data')
        plt.plot(x[1], y[1], 'g-', label = 'Predicted Data')
        if fut_pred>=seq_length:
            plt.axvline(x=self.times_predictions[seq_length-1], c='b', linestyle='--')
        plt.title('Deformation vs Time')
        plt.ylabel('Defs(cm)')
        plt.xlabel('Time(d)')
        plt.grid(True)
        plt.legend()
        fig_pred.savefig(fig_name)
        pass

    def plot_fit(self, fig_name, x, y, ind_test):
        fig_fit = plt.figure(4)
        fig_fit.clf()

        vline_substraction = self.times[ind_test]
        plt.axvline(x=vline_substraction, c='r', linestyle='--')

        plt.plot(x[0], y[0], 'r-', label = 'Raw Data')
        plt.plot(x[1], y[1], 'g-', label = 'Predicted Data')
        plt.title('Deformation vs Time')
        plt.ylabel('Defs(cm)')
        plt.xlabel('Time(d)')
        plt.grid(True)
        plt.legend()
        fig_fit.savefig(fig_name)

        pass

    def test_model(self, ind_test, seq_length, fut_pred, times, defsX, defsY, sc=None):
        # Load model
        self.load_model()

        ### Try to use train data
        try:
            ## Predictions over time

            test_inputs = np.zeros([fut_pred + 1, 1, seq_length, 1])
            #test_inputs[0] = dataX[-1].reshape(-1,seq_length,1).data.numpy()

            test_inputs[0] = defsX[ind_test].reshape(1,seq_length,1).data.numpy()

            time_step = np.absolute(times[0] - times[1])

            self.times = times
            self.times_predictions = (np.arange(0, (fut_pred+1)*time_step, time_step) +
                                      times[ind_test])

            for i in range(fut_pred):
                seq = torch.FloatTensor(test_inputs[i]).to(self.device)
                with torch.no_grad():
                    prediction = self.lstm(seq).data.cpu().numpy().item()
                    test_inputs[i+1] = np.append(test_inputs[i][0][1:], prediction).reshape([1,seq_length,1])


            data_predict = np.array([x.reshape(seq_length)[-1] for x in test_inputs]).reshape([-1,1])
            dataY_plot = defsY.data.cpu().numpy()

            data_predict = sc.inverse_transform(data_predict)
            dataY_plot = sc.inverse_transform(dataY_plot)

            # Coefficient of determination (R**2 score)
            pred_r2_score = r2s(dataY_plot[ind_test:ind_test+(fut_pred+1)], data_predict)
            print('Predicted - R^2 Score: ', pred_r2_score)

            self.plot_predict(self.current + "/defs_vs_times_pred" + self.params_name +
                              "_r2_" + str(round(pred_r2_score,3)) + ".jpg",
                              [self.times[ind_test:ind_test+(fut_pred+1)],self.times_predictions],
                              [dataY_plot[ind_test-1:ind_test-1+(fut_pred+1)],data_predict],
                              seq_length, fut_pred)
            ## Fitting whole model

            train_predict = self.lstm(defsX.to(self.device))    # Should be the same length as dataX
                                                                # but in a delayed window by 1 time
                                                                # (prediction) ==> last value

            data_predict = train_predict.data.cpu().numpy()
            dataY_plot = defsY.data.cpu().numpy()

            data_predict = sc.inverse_transform(data_predict)
            dataY_plot = sc.inverse_transform(dataY_plot)

            # Coefficient of determination (R**2 score)
            all_r2_score = r2s(dataY_plot, data_predict)
            print('All - R^2 Score: ', all_r2_score)

            self.plot_fit(self.current + "/defs_vs_times_fit" + self.params_name +
                          "_r2_" + str(round(all_r2_score,3)) + ".jpg",
                         [self.times, self.times],
                         [dataY_plot, data_predict],
                         ind_test)

        ### Use test data
        except:
            ## Predictions over time

            test_inputs = np.zeros([fut_pred + 1, 1, seq_length, 1])
            #test_inputs[0] = dataX[-1].reshape(-1,seq_length,1).data.numpy()

            test_inputs[0] = defsX.reshape(1,seq_length,1)

            time_step = np.absolute(times[0] - times[1])

            self.times = times
            self.times_predictions = (np.arange(0, (fut_pred+1)*time_step, time_step) +
                                      times[ind_test])

            for i in range(fut_pred):
                seq = torch.FloatTensor(test_inputs[i]).to(self.device)
                with torch.no_grad():
                    prediction = self.lstm(seq).data.cpu().numpy().item()
                    test_inputs[i+1] = np.append(test_inputs[i][0][1:], prediction).reshape([1,seq_length,1])


            data_predict = np.array([x.reshape(seq_length)[-1] for x in test_inputs]).reshape([-1,1])
            dataY_plot = defsY  # Pre-prediction deformation

            data_predict = sc.inverse_transform(data_predict)
            dataY_plot = sc.inverse_transform(dataY_plot)

            self.plot_predict(self.current + "/defs_vs_times_pred" + self.params_name + ".jpg",
                              [self.times[-1],self.times_predictions],
                              [dataY_plot,data_predict],
                              seq_length, fut_pred)
