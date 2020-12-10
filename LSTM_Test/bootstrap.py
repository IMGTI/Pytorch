import numpy as np
import os

### Set RNG seeds

seed = 55
np.random.seed(seed=seed)

# Main data folder
data_folder = 'datos/1_All_data'

# Sub data data folders
data_folders = np.array(os.listdir(data_folder))
np.random.shuffle(data_folders)

num_epochs = 500
for data_file in data_folders:
    # Call main.py
    os.system('python .\main.py -F -r ' + data_folder + '/' + data_file + ' -n ' + str(num_epochs))
