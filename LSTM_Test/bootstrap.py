import numpy as np
import os
import sys
import getopt
from pandas import ExcelWriter
import pandas as pd
from tqdm import tqdm

### Set RNG seeds

seed = 55
np.random.seed(seed=seed)

### Parse line arguments
def arg_parser(argv):
    mode = 'Train'
    aug = False
    try:
        opts, args = getopt.getopt(argv,"hm:a:",["mode=", "augment="])
    except getopt.GetoptError:
        print('argparser.py -m <Train/Test>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('------------ Commands -------------')
            print('-m: Mode (Train or Test)')
            print('-a: Augment Data (True or False)')
            print('-h: Show help')
            print('--------- Example usage ----------')
            print('For training:')
            print('argparser.py -m Train -a <True/False>')
            print('For testing:')
            print('argparser.py -m Test -a <True/False>')
            sys.exit()
        elif opt in ("-m", "--mode"):
            mode = arg
        elif opt in ("-a", "--augment"):
            if arg=='True':
                aug = True
            else:
                aug = False
    return mode, aug

if __name__ == "__main__":
    mode, aug = arg_parser(sys.argv[1:])

### Data augmentation

def augment_data(file_name, source_folder_name, target_folder_name):
    # Create folder if does not exist and copy original file into it
    try:
        os.mkdir(target_folder_name)
    except:
        pass

    # Read figures from excel
    data = pd.read_excel(source_folder_name + '/' + file_name, usecols=[0,1], names=['times', 'defs'])

    # Copy file from source to target
    if file_name not in os.listdir(target_folder_name):
        writer = ExcelWriter(target_folder_name + '/' + file_name[:-5] + '.xlsx',  datetime_format='dd-mm-yy hh:mm')
        data.to_excel(writer, index=False)
        writer.save()

    # Scale with random factor
    sc_coef = np.random.random()

    data['defs'] = sc_coef*np.array(data['defs'])

    # Create new excel with figures

    writer = ExcelWriter(target_folder_name + '/' + file_name[:-5] + '_sc_' + str(sc_coef) + '.xlsx',  datetime_format='dd-mm-yy hh:mm')
    '''

    # Invert deformation
    data['defs'] = (-1.0)*np.array(data['defs'])  # num_aug must be 1

    # Create new excel with figures

    writer = ExcelWriter(target_folder_name + '/' + file_name[:-5] + '_invert.xlsx',  datetime_format='dd-mm-yy hh:mm')
    '''
    data.to_excel(writer, index=False)
    writer.save()
    pass


# Main data folder
data_folder = '../../Datos_Radares/1_All_data'
augmented_data_folder = '../../Datos_Radares/1_All_data_augmented'

if aug==True:
    # Number of augmented files
    num_aug = 1#10
    # Sub data folders
    data_folders = np.array(os.listdir(data_folder))
    # Folder's names
    source_folder_name = data_folder
    target_folder_name = augmented_data_folder
    # Agument data
    print('Beginning Data Augmentation...')
    for file_name in tqdm(data_folders, total=len(data_folders)):
        ind = 1
        while ind<=num_aug:
            augment_data(file_name, source_folder_name , target_folder_name)
            ind += 1
    data_folder = target_folder_name
    data_folders = np.array(os.listdir(data_folder))
else:
    print('Using Augmented Data Folder...')
    data_folder = augmented_data_folder
    data_folders = np.array(os.listdir(data_folder))

# Shuffle files in data folders
np.random.shuffle(data_folders)

if mode=='Train':
    print('Initializing Training Mode...')
    num_epochs = 9999
    for data_file in tqdm(data_folders, total=len(data_folders)):
        # Call main.py
        os.system('python .\main.py -F -r ' + data_folder + '/' + data_file + ' -n ' + str(num_epochs))

else:
    print('Initializing Testing Mode...')
    for data_file in tqdm(data_folders, total=len(data_folders)):
        # Call main.py
        os.system('python .\main.py -t False -e ' + data_folder + '/' + data_file)
