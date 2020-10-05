### Define the hyperparameters ###

# Net parameters
num_epochs = 2000#200#300#2000
learning_rate = 0.001#0.001#0.01
input_size = 1
batch_size = 1  # Unused variable
hidden_size = 100#10#2
num_layers = 1

num_classes = 1


# Data parameters
seq_length = 12#1000#4  # Train Window
                        # 1h = 12
                        # 5min = 1

train_size = -100#int(len(y) * 0.67)
test_size = -100#len(y) - train_size  # Unused variable

fut_pred = 12#100  # Number of predictions

dropout = 0.05#0.05

# Parameters in name for .jpg files
params_name = ('_e' + str(num_epochs) +
               '_lr' + str(learning_rate) +
               '_b' + str(batch_size) +
               '_i' + str(input_size) +
               '_n' + str(num_layers) +
               '_h' + str(hidden_size) +
               '_o' + str(num_classes) +
               '_trw' + str(seq_length) +
               '_drp' + str(drop))

# Create directory for each run and different hyperparameters

current = dt.datetime.now().strftime("%d_%m_%Y") + '/' + dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

# Create new directory for each run

# Create directory
try:
    os.mkdir(dt.datetime.now().strftime("%d_%m_%Y"))
    os.mkdir(current)
except:
    try:
        os.mkdir(current)
    except:
        pass

# Path for state dictionary to save model's weights and parameters
state_dict_path = 'state_dict'

### SELECT DEVICE ###

# Send net to GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
