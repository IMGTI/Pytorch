import pandas as pd
import numpy as np
from pandas import ExcelWriter

def ext_mounts(data):
    # Divide data per mounts (when def is 0, create another dataset)
    ind_data = np.where(data.iloc[:,1].values==0)[0]
    datasets = {}
    for ind, val in enumerate(ind_data):
        if ind!=(len(ind_data)-1):
            datasets[ind] = data.iloc[ind_data[ind]:ind_data[ind+1]].reset_index(drop=True)
        else:
            datasets[ind] = data.iloc[ind_data[ind]:].reset_index(drop=True)
    return datasets

def ext_data(file_name):
    data_wo_mounts = True
    ind = 1
    while True:
        try:
            # Read figures from excel
            data = pd.read_excel(file_name, usecols=[0,ind], names=['times', 'defs'])

            # Extract mounts (erase artifacts)
            if data_wo_mounts:
                datasets = ext_mounts(data)

                # If datasets is empty, then there is no data, so break
                if datasets=={}:
                    break

                for ind_data, data in datasets.items():
                    # Create new excel with figures

                    writer = ExcelWriter(file_name[:-5] + '_fig_' + str(ind) + '_data_' + str(ind_data) + '.xlsx',  datetime_format='dd-mm-yy hh:mm')

                    data.to_excel(writer, index=False)
                    writer.save()
            else:
                if np.isnan(data['defs'].values[0]) or data['defs'].values[0]=='':
                    break
                writer = ExcelWriter(file_name[:-5] + '_fig_' + str(ind) + '.xlsx',  datetime_format='dd-mm-yy hh:mm')

                data.to_excel(writer, index=False)
                writer.save()

            ind += 1
        except:
            break

raw_input = input('>>> Nombre del archivo: ')
ext_data(raw_input)
