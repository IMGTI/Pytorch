import numpy as np
import os
import sys
import getopt
from pandas import ExcelWriter

### Set RNG seeds

seed = 55
np.random.seed(seed=seed)

### Parse line arguments
def arg_parser(argv):
    mode = 'Train'
    try:
        opts, args = getopt.getopt(argv,"hm:",["mode="])
    except getopt.GetoptError:
        print('argparser.py -m <Train/Test>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('------------ Commands -------------')
            print('-m: Mode (Train or Test)')
            print('-h: Show help')
            print('--------- Example usage ----------')
            print('For training:')
            print('argparser.py -m Train')
            print('For testing:')
            print('argparser.py -m Test')
            sys.exit()
        elif opt in ("-m", "--mode"):
            mode = arg

    return mode

if __name__ == "__main__":
    mode = arg_parser(sys.argv[1:])

### Data augmentation

def augment_data(file_name, source_folder_name, target_folder_name):
    # Create folder if does not exist and copy original file into it
    try:
        os.mkdir(target_folder_name)
        os.system("cp " + source_folder_name + "/" + file_name +
                  " " + target_folder_name + "/" + file_name)
    except:
        pass
    # Read figures from excel
    data = pd.read_excel(source_folder_name + '/' + file_name, usecols=[0,ind], names=['times', 'defs'])

    # Scale with random coefficient
    sc_coef = np.random.random()

    data['defs'] *= sc_coef

    # Create new excel with figures

    writer = ExcelWriter(source_folder_name + '/' + file_name[:-5] + '_sc_' + str(sc_coef) + '.xlsx',  datetime_format='dd-mm-yy hh:mm')

    data.to_excel(writer, index=False)
    writer.save()
    pass


# Main data folder
data_folder = 'datos/1_All_data'

# Sub data folders
data_folders = np.array(os.listdir(data_folder))

if mode=='Train':
    # Folder's names
    source_folder_name = data_folder
    target_folder_name = 'datos/1_All_data_augmented'
    # Agument data
    for file_name in data_folders:
        ind = 1
        while ind<=5:
            augment_data(file_name, source_folder_name , target_folder_name)
            ind += 1
    data_folders = np.array(os.listdir(target_folder_name))

# Shuffle files in data folders
np.random.shuffle(data_folders)

if mode=='Train':
    print('Initializing Training Mode...')
    num_epochs = 9999
    for data_file in data_folders:
        # Call main.py
        os.system('python .\main.py -F -r ' + data_folder + '/' + data_file + ' -n ' + str(num_epochs))

else:
    print('Initializing Testing Mode...')
    for data_file in data_folders:
        # Call main.py
        os.system('python .\main.py -t False -e ' + data_folder + '/' + data_file)
