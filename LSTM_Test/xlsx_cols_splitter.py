import pandas as pd
import numpy as np
from pandas import ExcelWriter

def ext_data(file_name):
    ind = 1
    while True:
        try:
            # Read figures from excel
            data = pd.read_excel(file_name, usecols=[0,ind], names=['times', 'defs'])
            if np.isnan(data['defs'].values[0]) or data['defs'].values[0]=='':
                break

            # Create new excel with figures

            writer = ExcelWriter(file_name + '_fig_' + str(ind) + '.xlsx',  datetime_format='dd-mm-yy hh:mm')

            data.to_excel(writer, index=False)
            writer.save()


            ind += 1
        except:
            break

raw_input = input('>>> Nombre del archivo: ')
ext_data(raw_input)
