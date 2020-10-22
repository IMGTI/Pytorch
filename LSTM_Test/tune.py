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
import getopt
import sys

### Parse line arguments
def arg_parser(argv):
    try:
        opts, args = getopt.getopt(argv,"hn:",["nsamples="])
    except getopt.GetoptError:
        print('argparser.py -n <number_samples>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('argparser.py -n <number_samples>')
            sys.exit()
        elif opt in ("-n", "--nsamples"):
            num_samples = int(arg)
    return num_samples

if __name__ == "__main__":
    num_samples = arg_parser(sys.argv[1:])
else:
    num_samples = 10

# Send to GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(data_dir):
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
        def mov_avg(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)

        # Apply Moving average

        N_avg = 2#5#2     # 2 para hacer una linea recta (tendencia) y al menos
                          # 5 puntos para tendencia valida (entonces con N_avg=2
                          # se logran 2-3 smooth ptos por cada 5)

        times = mov_avg(times, N_avg)
        defs = mov_avg(defs, N_avg)
        return times, defs

    fig_num = 1
    file = 'Figura_de_control_desde_feb_fig' + str(fig_num) + '.xlsx'

    times, defs = ext_data(os.path.abspath(data_dir + '/' + file))
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
    times, defs = load_data(data_dir)
    defsX, defsY, times_dataY, time_step = treat_data(times, defs, config["sl"])

    # Initialize model
    lstm = LSTM(1,1,config["hs"],config["nl"],0, False)
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
    if config["bs"]==-1:
        for epoch in range(10):
            optimizer.zero_grad()

            outputs = lstm(defsX.to(device))

            # Obtain the value for the loss function
            loss = criterion(outputs.to(device), defsY.to(device))

            loss4report = loss.clone()

            loss.backward()

            optimizer.step()

            # Save model
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((lstm.state_dict(), optimizer.state_dict()), path)

            # Report loss to tune
            tune.report(loss=loss4report.detach().cpu().numpy())
    else:
        batches = []
        ind = 0
        while True:
            try:
                batches.append({'defsX':torch.index_select(defsX, 0, torch.tensor(np.int64(np.arange(ind,ind+config["bs"],1)))),
                                'defsY':torch.index_select(defsY, 0, torch.tensor(np.int64(np.arange(ind,ind+config["bs"],1))))})
                ind += config["bs"]
            except:
                break
        for epoch in range(10):
            for batch in batches:
                optimizer.zero_grad()

                outputs = lstm(batch['defsX'].to(device))

                # Obtain the value for the loss function
                loss = criterion(outputs.to(device), batch['defsY'].to(device))

                loss4report = loss.clone()

                loss.backward()

                optimizer.step()

            # Save model
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((lstm.state_dict(), optimizer.state_dict()), path)

            # Report loss to tune
            tune.report(loss=loss4report.detach().cpu().numpy())
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
    data_dir = os.path.abspath(os.getcwd())
    # Configuration for raytune
    config = {
              "hs": tune.sample_from(lambda _: np.random.randint(1, 10)),
              "nl": tune.sample_from(lambda _: np.random.randint(1, 4)),
              "sl": tune.sample_from(lambda _: np.random.randint(1,100)),#(1, 288)),
              "bs": tune.sample_from(lambda _: np.random.randint(1,1000)),
              "lr": tune.loguniform(1e-4, 1e-1)
              }

    scheduler = ASHAScheduler(
                              metric="loss",
                              mode="min",
                              max_t=max_num_epochs,
                              grace_period=1,
                              reduction_factor=2)
    reporter = CLIReporter(
                           metric_columns=["loss", "training_iteration"])
    result = tune.run(
                      partial(train_model, data_dir=data_dir),
                      resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
                      config=config,
                      num_samples=num_samples,
                      scheduler=scheduler,
                      progress_reporter=reporter)

    df_result = result.results_df
    loss_result = df_result['loss'].to_numpy()
    ind_min_loss = np.argmin(loss_result)
    min_loss = np.min(loss_result)

    min_loss_trial = df_result.iloc[ind_min_loss]
    best_config = [df_result.iloc[ind_min_loss]['loss'],
                   df_result.iloc[ind_min_loss]['config.hs'],
                   df_result.iloc[ind_min_loss]['config.nl'],
                   df_result.iloc[ind_min_loss]['config.sl'],
                   df_result.iloc[ind_min_loss]['config.bs'],
                   df_result.iloc[ind_min_loss]['config.lr']]
    print('Best configuration parameters:')
    print('------------------------------')
    print(' Loss = ', best_config[0], '\n',
          'Hidden Size = ', best_config[1], '\n',
          'Number of layers = ', best_config[2], '\n',
          'Sequence length = ', best_config[3], '\n',
          'Batch Size = ', best_config[4], '\n',
          'Learning rate = ', best_config[5])

    '''
    NOT WORKING

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = LSTM(1,1,best_trial.config["hs"],best_trial.config["nl"],0)
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
    '''
if __name__ == "__main__":
    hyp_tune(num_samples=num_samples, max_num_epochs=10, gpus_per_trial=1)
