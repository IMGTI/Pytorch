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
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from model import LSTM
from sklearn.metrics import r2_score as r2s
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import joblib
import datetime as dt
import pandas as pd

# Send to GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(data_dir=""):
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

    def data_smooth(times, defs):
        # Apply Moving average

        N_avg = 2#5#2     # 2 para hacer una linea recta (tendencia) y al menos
                          # 5 puntos para tendencia valida (entonces con N_avg=2
                          # se logran 2-3 smooth ptos por cada 5)

        times = self.smooth(times, N_avg)
        defs = self.smooth(defs, N_avg)
        return times, defs

    fig_num = 1
    file = 'Figura_de_control_desde_feb_fig' + str(fig_num) + '.xlsx'

    times, defs = ext_data(file)
    times, defs = data_smooth(times, defs)

    return times, defs

def treat_data(times, defs, seq_length):
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
            data_sc = scaler.fit_transform(data)
        except:
            print('Scaler save file not found. Probably due to not trained model. ')

            scaler = MinMaxScaler()
            data_sc = scaler.fit_transform(data)

            # Save scaler for later use in test
            joblib.dump(scaler, sc_filename)
        return data_sc
    # Load data into sequences
    training_set = defs

    training_data = scaling(training_set)

    # Treat data
    x, y = sliding_windows(training_data, seq_length)

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x)))
    trainY = Variable(torch.Tensor(np.array(y)))

    # Times according with dataX and dataY dimensions
    time_step = np.absolute(times[0] - times[1])
    times_dataY = (times + (seq_length*time_step))[:-seq_length-1]

    return dataX, dataY, times_dataY, time_step

def train_model(config, checkpoint_dir="", data_dir=""):
    # Load data
    times, defs = load_data()
    defsX, defsY, times_dataY, time_step = treat_data(times, defs, config["sl"])

    # Initialize model
    lstm = LSTM(1,1,config["hs"],0)
    # Send model to device
    lstm.to(device)

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=config["lr"])

    # load model
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        lstm.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(defsX.to(device))
        optimizer.zero_grad()

        # Obtain the value for the loss function
        loss = criterion(outputs.to(device), defsY.to(device))

        loss.backward()

        optimizer.step()

        # Save model
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((lstm.state_dict(), optimizer.state_dict()), path)

        # Report loss to tune
        tune.report(loss=loss)
    pass

def test_accuracy(seq_length, model, device="cpu"):
    fut_pred = 12
    ind_test = -100

    times, defs = load_data()
    defsX, defsY, times_dataY, time_step = treat_data(times, defs, config["sl"])

    test_inputs = np.zeros([fut_pred + 1, 1, seq_length, 1])
    #test_inputs[0] = dataX[-1].reshape(-1,seq_length,1).data.numpy()

    test_inputs[0] = defsX[ind_test].reshape(1,seq_length,1)

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[i]).to(device)
        with torch.no_grad():
            prediction = model(seq).data.cpu().numpy().item()
            test_inputs[i+1] = np.append(test_inputs[i][0][1:], prediction).reshape([1,seq_length,1])

    data_predict = np.array([x.reshape(seq_length)[-1] for x in test_inputs]).reshape([-1,1])
    data = defsY  # Pre-prediction deformation

    data_predict = sc.inverse_transform(data_predict)
    data = sc.inverse_transform(dataY_plot)[ind_test:ind_test+(fut_pred+1)]

    # Test predicted model output with data
    r2s = r2s(data, data_predict)
    return r2s

def hyp_tune(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = ""
    # Configuration for raytune
    config = {
              "hs": tune.sample_from(lambda _: 2 ** np.random.randint(1, 6)),
              "nl": tune.uniform(1, 4),
              "sl": tune.uniform(1, 288),
              "lr": tune.loguniform(1e-4, 1e-1),
              "batch_size": tune.choice([2, 4, 8, 16])
              }
    scheduler = ASHAScheduler(
                              metric="loss",
                              mode="min",
                              max_t=max_num_epochs,
                              grace_period=1,
                              reduction_factor=2)
    reporter = CLIReporter(
                           # parameter_columns=["l1", "l2", "lr", "batch_size"],
                           metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
                      partial(train_model, data_dir=data_dir),
                      resources_per_trial={"cpu": 6, "gpu": gpus_per_trial},
                      config=config,
                      num_samples=num_samples,
                      scheduler=scheduler,
                      progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = LSTM(1,1,best_trial.config["hs"],0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trial.config["sl"], best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))

if __name__ == "__main__":
   hyp_tune(num_samples=1, max_num_epochs=10, gpus_per_trial=0)
