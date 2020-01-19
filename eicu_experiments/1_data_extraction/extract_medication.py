eicu_path = '/scratch/wiensj_root/wiensj/shared_data/datasets/eicu-2.0/'
save_path = '/scratch/wiensj_root/wiensj/shared_data/FIDDLE_project/extracted/'

import pandas as pd
import numpy as np
from tqdm import tqdm


config = {
    'n_rows': {
        'medication': 7_301_853,
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


fname = 'medication'
df_M = []
for i, df in enumerate(_read_events(fname, [], chunksize=100000)):
    # Remove unknow drug name or drug seqnum
    df['drughiclseqno'] = df['drughiclseqno'].astype('Int64')
    df = df.dropna(subset=['drugname', 'drughiclseqno'], how='all')
    
    # Combine drug name and ID
    df.loc[:, 'drugnameid'] = df[['drugname', 'drughiclseqno']].apply(
        lambda x: '{}|{}'.format(x[0], x[1]), axis=1)
    
    df = df.rename(columns={'patientunitstayid': 'ID', 'drugstartoffset': 't'})
    df = df.set_index([
        'ID', 't', 'drugnameid'
    ])[['dosage', 'routeadmin', 'frequency']]
    
    df.columns.name = 'property'
    df = df.stack()
    df.name = 'variable_value'
    df = df.reset_index()
    
    df['variable_name'] = df[['drugnameid', 'property']].apply(lambda x: '|'.join(x), axis=1)
    df['variable_value'] = pd.to_numeric(df['variable_value'], errors='ignore')
    df = df[['ID', 't', 'variable_name', 'variable_value']]
    
    df = df.reset_index(drop=True)
    df_M.append(df)

df_out = pd.concat(df_M, ignore_index=True)
try:
    df_out.to_parquet(save_path + '{}.parquet'.format(fname), index=False)
except:
    df_out.to_pickle(save_path + '{}.pickle'.format(fname))
