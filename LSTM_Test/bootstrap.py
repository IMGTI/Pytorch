import numpy as np
import os
import sys
import getopt

### Set RNG seeds

seed = 55
np.random.seed(seed=seed)

### Parse line arguments
def arg_parser(argv):
    mode = 'Train'
    try:
        opts, args = getopt.getopt(argv,"hm",["mode="])
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


# Main data folder
data_folder = 'datos/1_All_data'

# Sub data data folders
data_folders = np.array(os.listdir(data_folder))
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
