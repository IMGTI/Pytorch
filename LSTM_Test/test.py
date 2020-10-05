import numpy as np
import matplotlib.pyplot as plt
import torch
import datetime as dt
from model import LSTM
from main import (state_dict_path, num_classes, input_size, hidden_size, num_layers,
                  dropout, fut_pred, seq_length)

class Test(self):
    def __init__(self):
        # Initialize the model
        self.lstm = LSTM(num_classes, input_size, hidden_size, num_layers, dropout)
        pass

    def load_model(self):
        # Load state dict of model
        try:
            checkpoint = torch.load(state_dict_path)
            self.lstm.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            self.lstm.eval()
        except:
            print('State dict(s) missing')
        pass

    def test_model(self, times, defs):
        # Test predictions over time

        test_inputs = np.zeros([fut_pred + 1, 1, seq_length, 1])

        ind_test = -100#5000#1000#len(dataX)-1
        #test_inputs[0] = dataX[-1].reshape(-1,seq_length,1).data.numpy()

        test_inputs[0] = dataX[ind_test].reshape(-1,seq_length,1).data.numpy()

        time_step = np.absolute(times[0] - times[1])

        times_dataY = (times + (seq_length*time_step))[:-seq_length-1]  # Times according with dataX and dataY dimensions

        times_predictions = (np.arange(0, (fut_pred+1)*time_step, time_step) +
                             times_dataY[ind_test])

        for i in range(fut_pred):
            seq = torch.FloatTensor(test_inputs[i]).to(device)
            #print(seq)
            with torch.no_grad():
                prediction = lstm(seq).data.cpu().numpy().item()
                test_inputs[i+1] = np.append(test_inputs[i][0][1:], prediction).reshape([1,seq_length,1])


        #print(test_inputs)

        data_predict = np.array([x.reshape(seq_length)[-1] for x in test_inputs]).reshape([-1,1])
        dataY_plot = dataY.data.cpu().numpy()

        data_predict = sc.inverse_transform(data_predict)
        dataY_plot = sc.inverse_transform(dataY_plot)

        #print(data_predict)

        fig2 = plt.figure(2)
        fig2.clf()

        #plt.plot(range(-train_size), dataY_plot[train_size:], 'r-', label = 'Raw Data')
        plt.plot(times_dataY[ind_test:ind_test+(fut_pred+1)], dataY_plot[ind_test-1:ind_test-1+(fut_pred+1)], 'r-', label = 'Raw Data')
        #plt.plot(range(len(data_predict)), data_predict, 'g-', label = 'Predicted Data')
        plt.plot(times_predictions, data_predict, 'g-', label = 'Predicted Data')
        if fut_pred>=seq_length:
            plt.axvline(x=times_predictions[seq_length-1], c='b', linestyle='--')
        plt.title('Deformation vs Time')
        plt.ylabel('Defs(cm)')
        plt.xlabel('Time(d)')
        plt.grid(True)
        plt.legend()
        fig2.savefig(current + "/defs_vs_times_pred" + params_name + ".jpg")
        data_predict2 = data_predict
        pass

### TESTING ###

# Test predctions over time

test_inputs = np.zeros([fut_pred + 1, 1, seq_length, 1])

ind_test = -100#5000#1000#len(dataX)-1
#test_inputs[0] = dataX[-1].reshape(-1,seq_length,1).data.numpy()

test_inputs[0] = dataX[ind_test].reshape(-1,seq_length,1).data.numpy()

time_step = np.absolute(times[0] - times[1])

times_dataY = (times + (seq_length*time_step))[:-seq_length-1]  # Times according with dataX and dataY dimensions

times_predictions = (np.arange(0, (fut_pred+1)*time_step, time_step) +
                     times_dataY[ind_test])

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[i]).to(device)
    #print(seq)
    with torch.no_grad():
        prediction = lstm(seq).data.cpu().numpy().item()
        test_inputs[i+1] = np.append(test_inputs[i][0][1:], prediction).reshape([1,seq_length,1])


#print(test_inputs)

data_predict = np.array([x.reshape(seq_length)[-1] for x in test_inputs]).reshape([-1,1])
dataY_plot = dataY.data.cpu().numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

#print(data_predict)

fig2 = plt.figure(2)
fig2.clf()

#plt.plot(range(-train_size), dataY_plot[train_size:], 'r-', label = 'Raw Data')
plt.plot(times_dataY[ind_test:ind_test+(fut_pred+1)], dataY_plot[ind_test-1:ind_test-1+(fut_pred+1)], 'r-', label = 'Raw Data')
#plt.plot(range(len(data_predict)), data_predict, 'g-', label = 'Predicted Data')
plt.plot(times_predictions, data_predict, 'g-', label = 'Predicted Data')
if fut_pred>=seq_length:
    plt.axvline(x=times_predictions[seq_length-1], c='b', linestyle='--')
plt.title('Deformation vs Time')
plt.ylabel('Defs(cm)')
plt.xlabel('Time(d)')
plt.grid(True)
plt.legend()
fig2.savefig(current + "/defs_vs_times_pred" + params_name + ".jpg")
data_predict2 = data_predict
# Test fitting model

lstm.eval()

train_predict = lstm(dataX.to(device))  # Should be the same length as dataX
                                        # but in a delayed window by 1 time
                                        # (prediction) ==> last value

data_predict = train_predict.data.cpu().numpy()
dataY_plot = dataY.data.cpu().numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

fig3 = plt.figure(3)
fig3.clf()

#vline_substraction = np.absolute(train_size)# - (seq_length + 1)
vline_substraction = times_dataY[ind_test]
plt.axvline(x=vline_substraction, c='r', linestyle='--')

plt.plot(times_dataY, dataY_plot, 'r-', label = 'Raw Data')
plt.plot(times_dataY, data_predict, 'g-', label = 'Predicted Data')
plt.title('Deformation vs Time')
plt.ylabel('Defs(cm)')
plt.xlabel('Time(d)')
plt.grid(True)
plt.legend()
fig3.savefig(current + "/defs_vs_times_pred_fitting" + params_name + ".jpg")
