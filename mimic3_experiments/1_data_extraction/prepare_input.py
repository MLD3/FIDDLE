"""
python prepare_input.py
"""
import argparse
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from config import parallel, data_path, ID_col, t_col, var_col, val_col

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--outcome', type=str, required=True)
    parser.add_argument('--T', type=float, required=True)
    parser.add_argument('--dt', type=float, required=True)
    args = parser.parse_args()

    outcome = args.outcome
    T = args.T
    dt = args.dt
    
    print('Preparing pipeline input for: outcome={}, T={}, dt={}'.format(outcome, T, dt))
    
    import pathlib
    pathlib.Path(data_path, 'features', 'outcome={},T={},dt={}'.format(outcome, T, dt)) \
           .mkdir(parents=True, exist_ok=True)

    # Load in study population
    population = pd.read_csv(data_path + 'population/{}_{}h.csv'.format(outcome, T)) \
                   .rename(columns={'ICUSTAY_ID': 'ID'}).set_index('ID')[[]]

    # Load in raw data (from prepare.py)
    with open(data_path + 'formatted/all_data.stacked.p', 'rb') as f:
        data = pickle.load(f)

    # Resample continuous, resolve duplicates (discrete & continuous)
    data = resolve_duplicates_discrete(data)
    data = filter_prediction_time(data, T)
    data = resample_continuous_events(data, T, dt)
    data = resolve_duplicates_continuous(data)
    
    # Combine all DataFrames into one
    df_data = pd.concat(data, axis='index', ignore_index=True)
    df_data = df_data.sort_values(by=[ID_col, t_col, var_col, val_col], na_position='first')

    # Filter by IDs in study population
    df_data = population.join(df_data.set_index('ID')).reset_index()
    assert set(df_data['ID'].unique()) == set(population.index)

    # Save
    df_data.to_pickle(data_path + 'features/outcome={},T={},dt={}/input_data.p'.format(outcome, T, dt))


################################
####    Helper functions    ####
################################

def print_header(*content, char='='):
    print()
    print(char * 80)
    print(*content)
    print(char * 80, flush=True)

def filter_prediction_time(data_in, T):
    """
    Filter each table in `data_in` by:
        - Removing records outside of the prediction window [0, T) hours

    `data_in` is a dict {
        TABLE_NAME: pd.DataFrame object,
    }
    """
    print_header('Filter by prediction time T={}'.format(T), char='-')
    filtered_data = {}
    for table_name in tqdm(sorted(data_in)):
        df = data_in[table_name]
        t_cols = t_cols = df.columns.intersection(['t', 't_start', 't_end']).tolist()

        # Focus on the prediction window of [0, T)
        if len(t_cols) == 1: # point
            if all(pd.isnull(df['t'])):
                pass
            else:
                df = df[(0 <= df['t']) & (df['t'] < T)].copy()
        elif len(t_cols) == 2: # range
            df = df[(0 <= df['t_end']) & (df['t_start'] < T)].copy()

        filtered_data[table_name] = df
    
    print('Done!')
    return filtered_data


def resample_continuous_events(data, T, dt):
    print_header('Resample continuous events, T={}, dt={}'.format(T, dt), char='-')
    for fname, df in sorted(data.items(), reverse=True):
        t_cols = df.columns.intersection(['t', 't_start', 't_end']).tolist()
        if len(t_cols) == 1: # point time
            continue
        else: # ranged time
            assert len(t_cols) == 2
            print(fname)
            df_out = []
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                t_start, t_end = row.t_start, row.t_end
                t_range = dt/2 + np.arange(max(0, (t_start//dt)*dt), min(T, (t_end//dt+1)*dt), dt)
                if len(t_range) == 0:
                    continue
                df_tmp = pd.concat(len(t_range) * [row], axis=1).T.drop(columns=['t_start', 't_end'])
                df_tmp['t'] = t_range
                df_out.append(df_tmp)
            df_out = pd.concat(df_out)[['ID', 't', 'variable_name', 'variable_value']]
            data[fname] = df_out
    return data


def resolve_duplicates_discrete(data):
    """
    Assume input format:
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    |  ID  |  t (or t_start + t_end)  |  variable_name  |  variable_value  |
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    """
    print_header('Resolve duplicated event records (discrete)', char='-')

    ### Chart events - duplicate rows
    print('*** CHARTEVENTS')
    print('    getting dups and ~dups')
    df = data['CHARTEVENTS']
    m_dups = df.duplicated(subset=['ID', 't', 'variable_name'], keep=False)
    dups = df[m_dups]
    dup_variables = dups['variable_name'].unique()
    all_dups = df[df['variable_name'].isin(dup_variables)]
    not_dups = df[~df['variable_name'].isin(dup_variables)]
    
    def _resolve_duplicates_impl(v, df_v):
        # Categorical variables
        # Map to different variable names with value 0/1
        if pd.to_numeric(df_v['variable_value'], errors='ignore').dtype == 'object':
            df_mask = df_v.copy()
            df_mask['variable_value'] = 1
            df_mask = df_mask.drop_duplicates(subset=['ID', 't', 'variable_name'])

            df_attr = df_v.copy()
            df_attr['variable_name'] = df_attr['variable_name'].str.cat(df_attr['variable_value'], sep=': ')
            df_attr['variable_value'] = 1

            df_v_out = pd.concat([df_mask, df_attr], ignore_index=True).sort_values(by=['ID', 't', 'variable_name'])
        else:
            # Numerical variables: vitals, '226871', '226873'
            # Adjust timestamps by a small epsilon
            eps = 1e-6
            df_v_out = df_v.copy()
            for d_i, d_g in df_v.groupby('ID'):
                df_v_out.loc[d_g.index, 't'] += eps * np.arange(len(d_g))
        return df_v_out
    
    if parallel:
        df_dedup = Parallel(n_jobs=20, verbose=10)(delayed(_resolve_duplicates_impl)(v, df_v) for v, df_v in all_dups.groupby('variable_name'))
    else:
        df_dedup = [_resolve_duplicates_impl(v, df_v) for v, df_v in tqdm(all_dups.groupby('variable_name'))]
    
    print('    concatenating results')
    df_out = pd.concat([not_dups, *df_dedup])
    
    assert sum(df_out.duplicated()) == 7 ##### Handle the case where the exact record is duplicated
    df_out.drop_duplicates(inplace=True)
    assert not any(df_out.duplicated(subset=['ID', 't', 'variable_name'], keep=False))
    data['CHARTEVENTS'] = df_out.sort_values(by=['ID', 't', 'variable_name']).reset_index(drop=True)

    ### Labs - duplicate rows
    # Adjust timestamps by a small epsilon
    print('*** LABEVENTS')
    df = data['LABEVENTS']
    df_lab = df.copy()
    eps = 1e-6
    m_dups = df_lab.duplicated(subset=['ID', 't', 'variable_name'], keep=False)
    df_lab_out = df_lab[m_dups].copy()
    for v, df_v in df_lab_out.groupby('variable_name'):
        for d_i, d_g in df_v.groupby('ID'):
            df_lab_out.loc[d_g.index, 't'] += eps * np.arange(len(d_g))

    df_lab_final = pd.concat([df_lab[~m_dups], df_lab_out])
    assert not any(df_lab_final.duplicated(subset=['ID', 't', 'variable_name'], keep=False))
    data['LABEVENTS'] = df_lab_final.sort_values(by=['ID', 't', 'variable_name']).reset_index(drop=True)
    
    print('Verifying no more duplicates...')
    for table_name, df_table in data.items():
        if 't' in df_table.columns:
            assert not any(df_table.duplicated(subset=['ID', 't', 'variable_name'])), '{} contains duplicate records'.format(table_name)
    
    return data

def resolve_duplicates_continuous(data):
    print_header('Resolve duplicated event records (continuous)', char='-')
    
    # Remove repeated procedure events since they are just 0/1 mask
    print('*** PROCEDUREEVENTS_MV')
    data['PROCEDUREEVENTS_MV'] = data['PROCEDUREEVENTS_MV'].drop_duplicates(keep='first')
    
    # Handle duplicated input events
    ## Add up rates/amounts
    ## Map routes to separate indicator variables
    print('*** INPUTEVENTS_MV')
    df = data['INPUTEVENTS_MV']
    m_dups = df.duplicated(subset=['ID', 't', 'variable_name'], keep=False)
    dups = df[m_dups]
    dup_variables = dups['variable_name'].unique()
    all_dups = df[df['variable_name'].isin(dup_variables)]
    not_dups = df[~df['variable_name'].isin(dup_variables)]
    
    def resolve_duplicates_inputevents(v, df_v):
        # InputRoute - categorical
        if pd.to_numeric(df_v['variable_value'], errors='ignore').dtype == 'object':
            df_mask = df_v.copy()
            df_mask['variable_value'] = 1
            df_mask = df_mask.drop_duplicates(subset=['ID', 't', 'variable_name'])

            df_attr = df_v.copy()
            df_attr['variable_name'] = df_attr['variable_name'].str.cat(df_attr['variable_value'], sep=': ')
            df_attr['variable_value'] = 1

            df_v_out = pd.concat([df_mask, df_attr], ignore_index=True).drop_duplicates().sort_values(by=['ID', 't', 'variable_name'])
        else:
            # Numerical variables
            if len(v) == 6: # plain ITEMID
                # just use 0/1 indicator
                df_v_out = df_v.drop_duplicates(keep='first')
            else: # Rate/Amout - add up numbers at the same time stamp
                df_v_out = df_v.groupby(['ID', 't', 'variable_name'])[['variable_value']].sum().reset_index()
        return df_v_out
    
    if parallel:
        df_dedup = Parallel(n_jobs=20, verbose=10)(delayed(resolve_duplicates_inputevents)(v, df_v) for v, df_v in all_dups.groupby('variable_name'))
    else:
        df_dedup = [resolve_duplicates_inputevents(v, df_v) for v, df_v in tqdm(all_dups.groupby('variable_name'))]
    df_out = pd.concat([not_dups, *df_dedup])
    assert not any(df_out.duplicated(subset=['ID', 't', 'variable_name'], keep=False))
    data['INPUTEVENTS_MV'] = df_out
    
    
    print('Verifying no more duplicates...')
    for table_name, df_table in data.items():
        if 't' in df_table.columns:
            assert not any(df_table.duplicated(subset=['ID', 't', 'variable_name'])), '{} contains duplicate records'.format(table_name)
    
    return data


if __name__ == '__main__':
    main()
