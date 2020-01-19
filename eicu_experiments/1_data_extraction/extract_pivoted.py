# python extract_pivoted.py vitalPeriodic
# python extract_pivoted.py vitalAperiodic

eicu_path = '/scratch/wiensj_root/wiensj/shared_data/datasets/eicu-2.0/'
save_path = '/scratch/wiensj_root/wiensj/shared_data/FIDDLE_project/extracted/'

import pandas as pd
import numpy as np
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()
fname = args.filename

config = {
    'n_rows': {
        'vitalPeriodic': 146_671_642,
        'vitalAperiodic': 25_075_074,
    }
}

def _read_events(fname, t_cols, chunksize):
    """
    A helper function to read csv in chunks
    Arguments:
        - fname is the file name (i.e INPUTEVENTS)
        - t_cols is a list that contains the names of the time columns that should be parsed
        - chunksize is the size of each chunk
    """
    n_rows = config['n_rows'][fname]
    with tqdm(desc=fname, total=(n_rows//chunksize+1)) as pbar:
        for df in pd.read_csv(eicu_path + '{}.csv'.format(fname), parse_dates=t_cols, chunksize=chunksize):
            pbar.update()
            yield df




df_V = []
for i, df in enumerate(_read_events(fname, [], chunksize=1000000)):
    df = df.iloc[:,1:].set_index(['patientunitstayid', 'observationoffset'])
    df.columns.name = 'variable_name'
    df = df.stack()
    df.name = 'variable_value'
    df = df.reset_index()
    df_V.append(df)
    if i % 20 == 0:
        df_out = pd.concat(df_V, ignore_index=True)
        df_out.to_parquet(save_path + '{}.parquet'.format(fname), index=False)

df_out = pd.concat(df_V, ignore_index=True)
df_out.columns = ['ID', 't', 'variable_name', 'variable_value']
df_out = df_out.groupby(['ID', 't', 'variable_name']).median().reset_index() # Drop duplicates and keep the median value
df_out.to_parquet(save_path + '{}.parquet'.format(fname), index=False)
