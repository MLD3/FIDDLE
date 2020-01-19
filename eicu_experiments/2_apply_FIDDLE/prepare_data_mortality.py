eicu_path = '/scratch/wiensj_root/wiensj/shared_data/datasets/eicu-2.0/' #read from here
data_path = '/scratch/wiensj_root/wiensj/shared_data/FIDDLE_project/' #save here

import pandas as pd
import numpy as np
from tqdm import tqdm

import pandas as pd
import pickle
import os

data_path = '/scratch/wiensj_root/wiensj/shared_data/FIDDLE_project/'
pop_path = 'population/mortality_48h.csv'

pop = pd.read_csv(data_path + pop_path)
T = 48.0
data = []

for i, filename in enumerate(reversed(sorted(os.listdir(data_path + 'extracted/')))):
    if filename.endswith(".parquet") or filename.endswith(".pickle"): 
        print('___', filename, '___', flush=True)
        
        if filename.endswith(".parquet"): df = pd.read_parquet(data_path + 'extracted/' + filename)
        else: df = pickle.load(open(data_path + 'extracted/' + filename, 'rb'))
    
        #subsetting population
        print('rows: ', df.shape[0])
        df = df[df.ID.isin(pop.ID)]
        print('rows after subsetting population: ', df.shape[0])
        
        #subsetting time
        df = df[((df.t >= 0) & (df.t < T*60)) | np.isnan(df.t)]
        print('rows after subsetting time: ', df.shape[0])
        
        df['variable_value'] = pd.to_numeric(df['variable_value'], errors='ignore')
        data.append(df)
        del df

data = pd.concat(data)
print(data.shape)
print(data.head())

print('Number of unique variable_names:', data['variable_name'].nunique())
print('Number of rows:', len(data))

# Remove duplicate rows and recording any duplicates and inconsistencies
data = data.drop_duplicates(subset=['ID', 't', 'variable_name'], keep='first')
data = data.sort_values(by=['ID', 't', 'variable_name'])
print('Number of rows after removing duplicate rows:', len(data))

data.to_csv(data_path + 'features/mortality/input_data.csv', index=False)
