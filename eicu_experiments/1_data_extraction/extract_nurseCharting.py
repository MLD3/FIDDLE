eicu_path = '/scratch/wiensj_root/wiensj/shared_data/datasets/eicu-2.0/'
save_path = '/scratch/wiensj_root/wiensj/shared_data/FIDDLE_project/extracted/'

import pandas as pd
import numpy as np
from tqdm import tqdm

config = {
    'n_rows': {
        'nurseCharting': 151_604_232,
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


fname = 'nurseCharting'
df_NC = []
for i, df in enumerate(_read_events(fname, [], chunksize=1000000)):
    df = df.drop(columns=[
        'nursingchartid', 
        'nursingchartentryoffset', 
    ])
    df = df.rename(columns={
        'patientunitstayid': 'ID',
        'nursingchartoffset': 't',
    })
    df['variable_name'] = df[[
        'nursingchartcelltypecat', 'nursingchartcelltypevallabel', 
        'nursingchartcelltypevalname'
    ]].apply(lambda x: '|'.join(x), axis=1)

    df['variable_value'] = pd.to_numeric(df['nursingchartvalue'], errors='ignore')

    df = df[['ID', 't', 'variable_name', 'variable_value']]
    df = df.reset_index(drop=True)
    df_NC.append(df)
    if i % 40 == 39:
        df_out = pd.concat(df_NC, ignore_index=True)
        try:
            df_out.to_parquet(data_path + '{}_{}.parquet'.format(fname, int(i//40)), index=False)
        except:
            df_out.to_pickle(data_path + '{}_{}.pickle'.format(fname, int(i//40)))
        df_NC = []

df_out = pd.concat(df_NC, ignore_index=True)
try:
    df_out.to_parquet(save_path + '{}_{}.parquet'.format(fname, int(i//40)), index=False)
except:
    df_out.to_pickle(save_path + '{}_{}.pickle'.format(fname, int(i//40)))