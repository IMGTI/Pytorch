import numpy as np
import matplotlib.pyplot as plt
import torch
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as r2s
from model_spc import CNN
import pandas as pd
from pandas import ExcelWriter

class Test(object):
    def __init__(self, batch_size, num_classes, input_size, hidden_size, num_layers, dropout,
                 bidirectional, state_dict_path, current, params_name, seed, tfile=''):
        # RNG Seed
        np.random.seed(seed)  # Numpy
        torch.manual_seed(seed)  # Pytorch

        # Save test file
        if tfile!='':
            self.test_file = tfile

        # Initialize the model
        self.lstm = LSTM(batch_size, num_classes, input_size, hidden_size, num_layers,
                         dropout, bidirectional, seed)
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
        plt.plot(x[1][0], y[1][0], 'b*', label = 'Start point (Raw)')
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

    def include_rw(self, ind):
        self.rev_rand = ind
        pass

    def add_pred_to_data(self, ind_source, new_data):
        new_times, new_defs = new_data

        # Read source file
        path_source = '../../Datos_Radares/1_All_data/' + self.test_file
        data_source = pd.read_excel(path_source, usecols=[0,1], names=['old_times', 'old_defs'])

        # Select data until index used for prediction
        data_source = data_source.iloc[:ind_source+1]

        # Add initial new time (final source time)
        new_times += (dt.datetime.timestamp(data_source['old_times'].iloc[-1]))/(24*60*60)  # Seconds to days

        # Format new time (remember times was in days, must be converted to seconds)
        new_times = np.array([dt.datetime.fromtimestamp(x*24*60*60) for x in new_times])

        # Create dictionaries to save as excel files

        # Prediction dictionary
        df = data_source.copy()
        df = df.iloc[:len(new_times)]
        df['old_times'], df['old_defs'] = new_times, new_defs
        # Append prediction dictionary to source data copy dictionary
        data_pred = data_source.copy()
        data_pred = data_pred.append(df)
        data_pred = data_pred.rename(columns={'old_times':'new_times', 'old_defs':'new_defs'})

        # Create excel file
        writer_orig = ExcelWriter(self.current + '/original.xlsx',  datetime_format='dd/mm/yyyy hh:mm')
        writer_pred = ExcelWriter(self.current + '/predicted.xlsx',  datetime_format='dd/mm/yyyy hh:mm')
        data_source.to_excel(writer_orig, sheet_name='old_defs', index=False)
        data_pred.to_excel(writer_pred, sheet_name='new_defs', index=False)
        writer_orig.save()
        writer_pred.save()
        pass

    def test_model(self, ind_test, seq_length, fut_pred, times, defsX, defsY, sc=None):
        # Load model
        self.load_model()

        # Try to reorder data if its randomized
        try:
            defsX, defsY = defsX[self.rev_rand], defsY[self.rev_rand]
        except:
            print('Data is not randomized!')

        ### Validation test
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
                    prediction, hidden = self.lstm(seq)
                    prediction = prediction.data.cpu().numpy().item()
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

            train_predict, hidden_train_predict = self.lstm(defsX.to(self.device))  # Should be the same length as dataX
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

        ### Data Test (Prediction)
        except:
            ## Predictions over time

            test_inputs = np.zeros([fut_pred + 1, 1, seq_length, 1])
            #test_inputs[0] = dataX[-1].reshape(-1,seq_length,1).data.numpy()

            test_inputs[0] = defsX.reshape(1,seq_length,1)

            time_step = np.absolute(times[0] - times[1])

            self.times = times
            self.times_predictions = (np.arange(0, (fut_pred+1)*time_step, time_step) +
                                      times[-1])

            for i in range(fut_pred):
                seq = torch.FloatTensor(test_inputs[i]).to(self.device)
                with torch.no_grad():
                    prediction, hidden = self.lstm(seq)
                    prediction = prediction.data.cpu().numpy().item()
                    test_inputs[i+1] = np.append(test_inputs[i][0][1:], prediction).reshape([1,seq_length,1])


            data_predict = np.array([x.reshape(seq_length)[-1] for x in test_inputs]).reshape([-1,1])
            dataY_plot = defsY  # Pre-prediction deformation

            data_predict = sc.inverse_transform(data_predict)
            dataY_plot = sc.inverse_transform(dataY_plot)

            self.plot_predict(self.current + "/defs_vs_times_pred" + self.params_name + ".jpg",
                              [np.hstack([self.times[:-1], self.times_predictions])[:len(dataY_plot)],self.times_predictions],
                              [dataY_plot,data_predict],
                              seq_length, fut_pred)

            ## Add predicted data to excel file
            new_data = (self.times_predictions.reshape(-1)[1:]-times[-1], data_predict.reshape(-1)[1:])  # Dont repeat first sample
            self.add_pred_to_data(ind_test, new_data)
