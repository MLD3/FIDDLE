"""
FIDDLE Preprocessing steps
1. Pre-filter
2. Transform
3. Post-filter
"""
from .helpers import *
import time
import json

def pre_filter(df, threshold, df_population, args):
    T = int(args.T)
    theta_1 = args.theta_1
    df_population = args.df_population
    
    # Remove rows not in population
    print('Remove rows not in population')
    df = df[df['ID'].isin(df_population.index)]
    
    # Remove rows with t outside of [0, T)
    print('Remove rows with t outside of [0, {}]'.format(T))
    df = df[pd.isnull(df[t_col]) | ((0 <= df[t_col]) & (df[t_col] < T))]
    
    # Data tables should not contain duplicate rows
    # Check for inconsistencies
    dups = df.duplicated(subset=[ID_col, t_col, var_col], keep=False)
    if any(dups):
        print(df[dups].head())
        raise Exception('Inconsistent values recorded')
    
    # Remove variables that occur too rarely as defined by the threshold
    print('Remove rare variables (<= {})'.format(threshold))
    
    ## Calculate overall occurrence rate of each variable based on IDs
    df_count = calculate_variable_counts(df, df_population) # (N x |var|) table of counts
    df_bool = df_count.astype(bool) # convert counts to boolean
    
    ## Keep variables that are recorded for more than threshold fraction of IDs
    variables_keep = df_bool.columns[df_bool.mean(axis=0) > threshold]
    df_out = df[df[var_col].isin(variables_keep)]
    assert set(variables_keep) == set(df_out[var_col].unique())
    
    variables = sorted(df_bool.columns)
    variables_remove = sorted(set(variables) - set(variables_keep))
    print('Total variables     :', len(variables))
    print('Rare variables      :', len(variables_remove))
    print('Remaining variables :', len(variables_keep))
    print('# rows (original)   :', len(df))
    print('# rows (filtered)   :', len(df_out))
    return df_out


def detect_variable_data_type(df_data, value_type_override, args):
    data_path = args.data_path
    print_header('*) Detecting value types', char='-')
    
    data_types = []
    df = df_data
    assert val_col in df.columns

    # Collect the unique values of each variable
    # values_by_variable: dict(variable_name -> [value1, value2, ...])
    d = df[[var_col, val_col]].drop_duplicates().sort_values(by=[var_col, val_col])
    values_by_variable = defaultdict(list)
    for n,v in zip(d[var_col], d[val_col]):
        values_by_variable[n].append(v)

    # Determine type of each variable
    for variable, values in sorted(values_by_variable.items()):
        # Manual override type in config
        if variable in value_type_override:
            data_types.append((variable, value_type_override[variable]))
            # Force categorical values to be a string
            if value_type_override[variable] == 'Categorical' and \
                any(is_numeric(v) for v in values if not pd.isnull(v)):
                m_var = df[var_col] == variable
                df.loc[m_var, val_col] = df.loc[m_var, val_col].apply(lambda s: '_' + str(s))
        else:
            if len(values) == 1 and pd.isnull(values[0]):
                data_types.append((variable, 'None'))
            elif all(is_numeric(v) for v in values if not pd.isnull(v)):
                data_types.append((variable, 'Numeric'))
            elif any(is_numeric(v) for v in values if not pd.isnull(v)):
                data_types.append((variable, 'Numeric + Categorical'))
            else:
                data_types.append((variable, 'Categorical'))
    
    df_types = pd.DataFrame(data_types, columns=['variable_name', 'value_type'])
    df_types[var_col] = df_types[var_col].astype(str)
    df_types = df_types.set_index(var_col)
    fpath = data_path + 'value_types.csv'
    df_types.to_csv(fpath, quoting=1)
    print('Saved as:', fpath)
    return df, df_types['value_type']


def split_by_timestamp_type(df):
    print_header('*) Separate time-invariant and time-dependent', char='-')
    
    variables_inv = df[pd.isnull(df[t_col])][var_col].unique() # Invariant variables have t = NULL
    df_time_invariant = df[df[var_col].isin(variables_inv)]
    df_time_series = df[~df[var_col].isin(variables_inv)]
    
    print('Variables (time-invariant):', len(variables_inv))
    print('Variables (time-dependent):', df[var_col].nunique() - len(variables_inv))
    print('# rows    (time-invariant):', len(df_time_invariant))
    print('# rows    (time-dependent):', len(df_time_series))
    return df_time_invariant, df_time_series


def process_time_invariant(df_data_time_invariant, args):
    data_path = args.data_path
    df_population = args.df_population
    theta_2 = args.theta_2
    
    print_header('2-A) Transform time-invariant data', char='-')
    dir_path = data_path + '/'
    start_time = time.time()

    ## Create Nxd^ table
    df_time_invariant = transform_time_invariant_table(df_data_time_invariant, df_population)
    print('Time elapsed: %f seconds' % (time.time() - start_time))

    ## Discretize
    s_all, s_all_feature_names = map_time_invariant_features(df_time_invariant, args)
    sparse.save_npz(dir_path + 's_all.npz', s_all)
    with open(dir_path + 's_all.feature_names.json', 'w') as f:
        json.dump(list(s_all_feature_names), f, sort_keys=True)
    print('Time elapsed: %f seconds' % (time.time() - start_time))

    print_header('3-A) Post-filter time-invariant data', char='-')
    
    ## Filter
    s, s_feature_names, s_feature_aliases = post_filter(s_all, s_all_feature_names, theta_2)
    print('Time elapsed: %f seconds' % (time.time() - start_time))
    
    ## Save output
    print()
    print('Output')
    print('s: shape={}, density={:.3f}'.format(s.shape, s.density))
    sparse.save_npz(dir_path + 's.npz', s)
    
    with open(dir_path + 's.feature_names.json', 'w') as f:
        json.dump(list(s_feature_names), f, sort_keys=True)
    with open(dir_path + 's.feature_aliases.json', 'w') as f:
        json.dump(s_feature_aliases, f, sort_keys=True)
    
    print('Total time: %f seconds' % (time.time() - start_time))
    print('', flush=True)
    return s, s_feature_names, s_feature_aliases


def process_time_dependent(df_data_time_series, args):
    data_path = args.data_path
    theta_2 = args.theta_2
    
    print_header('2-B) Transform time-dependent data', char='-')
    dir_path = data_path + '/'
    start_time = time.time()

    ## Create NxLxD^ table
    df_time_series, dtypes_time_series = transform_time_series_table(df_data_time_series, args)
    print('Time elapsed: %f seconds' % (time.time() - start_time))
    
    ## Map variables to features
    X_all, X_all_feature_names = map_time_series_features(df_time_series, dtypes_time_series, args)
    sparse.save_npz(dir_path + 'X_all.npz', X_all)
    with open(dir_path + 'X_all.feature_names.json', 'w') as f:
        json.dump(list(X_all_feature_names), f, sort_keys=True)
    print('Time elapsed: %f seconds' % (time.time() - start_time))
    
    ## Filter features
    print_header('3-B) Post-filter time-dependent data', char='-')
    print(X_all.shape, X_all.density)
    X, X_feature_names, X_feature_aliases = post_filter_time_series(X_all, X_all_feature_names, theta_2, args)
    print(X.shape, X.density)
    print('Time elapsed: %f seconds' % (time.time() - start_time))

    ## Save output
    print()
    print('Output')
    print('X: shape={}, density={:.3f}'.format(X.shape, X.density))
    sparse.save_npz(dir_path + 'X.npz', X)
    with open(dir_path + 'X.feature_names.json', 'w') as f:
        json.dump(list(X_feature_names), f, sort_keys=True)
    with open(dir_path + 'X.feature_aliases.json', 'w') as f:
        json.dump(X_feature_aliases, f, sort_keys=True)
    
    print('Total time: %f seconds' % (time.time() - start_time))
    print('', flush=True)
    return X, X_feature_names, X_feature_aliases


######
# Time-invariant routines
######
def transform_time_invariant_table(df_in, df_population):
    df_in = df_in.copy()
    
    # Recorded Value (np.nan if not recorded)
    df_value = pd.pivot_table(df_in, val_col, ID_col, var_col, 'last', np.nan)
    df_value = df_value.reindex(index=df_population.index, fill_value=np.nan)
    df_value.columns = [str(col) + '_value' for col in df_value.columns]
    
    print('(N \u00D7 ^d) table            :\t', df_value.shape)
    print('number of missing entries :\t', '{} out of {} total'.format(df_value.isna().sum().sum(), df_value.size))
    return df_value

def map_time_invariant_features(df, args):
    # Categorical -> binary features
    # Numeric -> binary/float-valued features
    if args.binarize:
        out = [smart_qcut_dummify(df[col], q=5, use_ordinal_encoding=use_ordinal_encoding) for col in df.columns]
        time_invariant_features = pd.concat(out, axis=1)
        feature_names_all = time_invariant_features.columns.values
        sdf = time_invariant_features.astype(pd.SparseDtype(int, fill_value=0))
        s_ = sparse.COO(sdf.sparse.to_coo())
    else:
        # Split a mixed column into numeric and string columns
        for col in df.columns:
            col_data = df[col]
            col_is_numeric = [is_numeric(v) for v in col_data if not pd.isnull(v)]
            if not all(col_is_numeric) and any(col_is_numeric): # have mixed type values
                numeric_mask = col_data.apply(is_numeric)
                df[col+'_str'] = df[col].copy()
                df.loc[~numeric_mask, col] = np.nan
                df.loc[numeric_mask, col+'_str'] = np.nan
        
        out = [smart_dummify_impute(df[col]) for col in df.columns]
        time_invariant_features = pd.concat(out, axis=1)
        feature_names_all = time_invariant_features.columns.values
        sdf = time_invariant_features.astype(pd.SparseDtype(float, fill_value=0))
        s_ = sparse.COO(sdf.sparse.to_coo())
    
    print()
    print('Output')
    print('s_all, binary features    :\t', s_.shape)
    return s_, feature_names_all

def post_filter(s_, s_feature_names_all, threshold):
    # Filter features (optional)
    assert s_.shape[1] == len(s_feature_names_all)
    feature_names_0 = s_feature_names_all
    s0 = s_.to_scipy_sparse()
    print('Original       :', len(feature_names_0))
    
    ## Remove nearly-constant features (with low variance)
    ## a binary feature is removed if =0 (or =1) for >th fraction of examples
    ## i.e., variance <= (th * (1 - th))
    sel_rare = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    s1 = sel_rare.fit_transform(s0)
    feature_names_1 = feature_names_0[sel_rare.get_support()]
    print('Nearly-constant:', len(feature_names_0) - len(feature_names_1))
    
    ## Keep only first of pairwise perfectly correlated features
    sel_corr = CorrelationSelector()
    s2 = sel_corr.fit_transform(s1)
    feature_names_2 = feature_names_1[sel_corr.get_support()]
    feature_aliases = sel_corr.get_feature_aliases(feature_names_1)
    print('Correlated     :', len(feature_names_1) - len(feature_names_2))
    
    s = sparse.COO(s2)
    feature_names = feature_names_2
    assert s.shape[1] == len(feature_names)
    
    return s, feature_names, feature_aliases


######
# Time-series routines
######
def func_encode_single_time_series(i, g, variables, variables_num_freq, T, dt, stats_functions, impute=True):
    try:
        assert g['ID'].nunique() == 1
        assert g['ID'].unique()[0] == i
        # non-frequent
        variables_non = sorted(set(variables) - set(variables_num_freq))
        df_j = pivot_event_table(g).reindex(columns=variables_non).sort_index()
        df_values_j = most_recent_values(df_j, variables, T, dt)
        df_out = df_values_j
        
        if len(variables_num_freq) > 0:
            # frequent
            # we're only producing mask, ffill, and statistics if the data is measured frequently enough
            df_i = pivot_event_table(g).reindex(columns=variables_num_freq).sort_index()
            mask_i = presence_mask(df_i, variables_num_freq, T, dt)
            delta_t_i = get_delta_time(mask_i)
            df_i = impute_ffill(df_i, variables_num_freq, T, dt, mask_i)
            df_stats_i = summary_statistics(df_i, variables_num_freq, stats_functions, T, dt)
            df_values_i = most_recent_values(df_i, variables, T, dt)
            if impute:
                check_imputed_output(df_values_i)
                check_imputed_output(df_stats_i)
        
            df_out = df_out.join([mask_i, delta_t_i, df_values_i, df_stats_i])
    except:
        print(i)
        raise Exception(i)
    return i, df_out

def transform_time_series_table(df_in, args):
    data_path = args.data_path
    theta_freq = args.theta_freq
    stats_functions = args.stats_functions
    N, L = args.N, args.L
    df_population = args.df_population
    parallel = args.parallel
    
    ## TODO: asserts shape of df_in

    # Determine all unique variable names
    variables = get_unique_variables(df_in)
    assert df_in[var_col].nunique() == len(variables)
    print('Total variables    :', len(variables))
    
    # Determine frequent variables -> we'll calculate statistics, mask, and delta_time only on these
    variables_num_freq = get_frequent_numeric_variables(df_in, variables, theta_freq, args)
    print('Frequent variables :', list(variables_num_freq))
    print('{} = {}'.format('M\u2081', len(variables_num_freq)))
    print('{} = {}'.format('M\u2082', len(variables) - len(variables_num_freq)))
    print('{} = {} {}'.format('k ', len(stats_functions), stats_functions))
    
    print()
    print('Transforming each example...')
    # Encode time series table for each patient
    grouped = list(df_in.groupby(ID_col))
    if parallel:
        out = dict(Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(func_encode_single_time_series)(i, g, variables, variables_num_freq, args.T, args.dt, args.stats_functions)
            for i, g in grouped[:N]
        ))
        
    else:
        out = dict(
            func_encode_single_time_series(i, g, variables, variables_num_freq, args.T, args.dt, args.stats_functions) 
            for i, g in tqdm(grouped[:N])
        )
    
    # Handle IDs not in the table
    df_original = list(out.values())[0]
    df_copy = pd.DataFrame().reindex_like(df_original)
    for i, j in df_original.dtypes.iteritems():
        if i.endswith('_mask'):
            assert j == bool
            df_copy[i] = False
            df_copy[i] = df_copy[i].astype(bool)
        if i.endswith('_delta_time'):
            df_copy[i] = 0
            df_copy[i] = df_copy[i].astype(int)
        if j == 'object':
            df_copy[i] = df_copy[i].astype('object')

    for ID in df_population.index.values[:N]:
        if ID not in out:
            out[ID] = df_copy.copy()

    out = {ID: out[ID] for ID in df_population.index.values[:N]}
    assert len(out) == N
    D_timeseries = out

    # check each example have identical LxD table structure
    ID0 = sorted(D_timeseries.keys())[0]
    df0 = D_timeseries[ID0]
    for ID, df_i in D_timeseries.items():
        pd.testing.assert_index_equal(df_i.index, df0.index)
        pd.testing.assert_index_equal(df_i.columns, df0.columns)

    D_timeseries = out
    D_ = len(list(D_timeseries.values())[0].columns)
    
    # (N*L)xD^ table
    ## Create MultiIndex of (ID, time_bin)
    index = sum([ 
        [(ID, t_) for t_ in list(df_.index)]
        for ID, df_ in sorted(D_timeseries.items()) 
    ], [])
    index = pd.Index(index)
    assert len(index) == N * L
    
    ## Assume all dataframes have the same columns, used after concatenation
    columns = list(sorted(D_timeseries.items())[0][1].columns)
    columns = np.array(columns)
    dtypes = sorted(D_timeseries.items())[0][1].dtypes
    
    ## Convert each df to a numpy array
    ## Concatenate **sorted** numpy arrays (faster than calling pd.concat)
    feature_values = [(ID, df_.to_numpy()) for ID, df_ in sorted(D_timeseries.items())]
    time_series = np.concatenate([feat_val[1] for feat_val in feature_values])
    assert time_series.shape == (len(index), len(columns))
    
    df_time_series = pd.DataFrame(data=time_series, index=index, columns=columns)
    
    # Print metadata
    print('DONE: Transforming each example...')
    ## Freq: Count missing entries using mask
    ts_mask = df_time_series[[col for col in df_time_series if col.endswith('_mask')]]
    ts_mask.columns = [col.replace('_mask', '') for col in ts_mask.columns]
    print('(freq) number of missing entries :\t', 
          '{} out of {}={} total'.format(
              (1-ts_mask).astype(int).sum().sum(), 
              '\u00D7'.join(str(i) for i in [N,L,ts_mask.shape[1]]), ts_mask.size))
    
    ## Freq: Count imputed entries using mask and dt
    ts_delta_time = df_time_series[[col for col in df_time_series if col.endswith('_delta_time')]]
    ts_delta_time.columns = [col.replace('_delta_time', '') for col in ts_delta_time.columns]
    
    imputed = (1-ts_mask).astype(bool) & (ts_delta_time > 0)
    print('(freq) number of imputed entries :\t', 
          '{}'.format(imputed.sum().sum(), ts_delta_time.size))
    imputed.sum().rename('count').to_csv(data_path + '/' + 'freq_imputed.csv')
    
    not_imputed = (1-ts_mask).astype(bool) & (ts_delta_time == 0)
    print('(freq) number of not imputed entries :\t', 
          '{}'.format(not_imputed.sum().sum(), ts_delta_time.size))
    not_imputed.sum().rename('count').to_csv(data_path + '/' + 'freq_not_imputed.csv')
    
    ## Non-Freq: Count missing entries
    non_freq_cols = sorted([c + '_value' for c in set(variables) - set(variables_num_freq)])
    non_freqs = df_time_series[non_freq_cols]
    print('(non-freq) number of missing entries :\t',
          '{} out of {}={} total'.format(
              non_freqs.isna().sum().sum(), 
              '\u00D7'.join(str(i) for i in [N,L,non_freqs.shape[1]]), non_freqs.size))
    
    print()
    print('(N \u00D7 L \u00D7 ^D) table :\t', (N, L, len(columns)))
    return df_time_series, dtypes

def map_time_series_features(df_time_series, dtypes, args):
    N, L = args.N, args.L
    
    df_time_series = df_time_series.dropna(axis='columns', how='all').sort_index()

    print('Discretizing features...')
    ts_mask = select_dtype(df_time_series, 'mask', dtypes)
    ts_mixed = select_dtype(df_time_series, '~mask', dtypes)
    assert len(ts_mixed.columns) + len(ts_mask.columns) == len(df_time_series.columns)
    ts_feature_mask = ts_mask.astype(int)
    ts_mixed_cols = [ts_mixed[col] for col in ts_mixed.columns]
    
    print()
    if args.binarize:
        dtype = int
        print('Processing', len(ts_mixed_cols), 'non-boolean variable columns...')

        print('    Binning numeric variables by quintile...')
        print('    Converting variables to binary features')
        if parallel:
            out = Parallel(n_jobs=n_jobs, verbose=10)( # Need to share global variables
                delayed(smart_qcut_dummify)(col_data, q=5, use_ordinal_encoding=use_ordinal_encoding) for col_data in ts_mixed_cols
            )
        else:
            out = [smart_qcut_dummify(col_data, q=5, use_ordinal_encoding=use_ordinal_encoding) for col_data in tqdm(ts_mixed_cols)]
    else:
        dtype = float
        df = ts_mixed.copy()
        
        # Split a mixed column into numeric and string columns
        for col in df.columns:
            col_data = df[col]
            col_is_numeric = [is_numeric(v) for v in col_data if not pd.isnull(v)]
            if not all(col_is_numeric) and any(col_is_numeric): # have mixed type values
                numeric_mask = col_data.apply(is_numeric)
                df[col+'_str'] = df[col].copy()
                df.loc[~numeric_mask, col] = np.nan
                df.loc[numeric_mask, col+'_str'] = np.nan
        
        ts_mixed_cols = [df[col] for col in df.columns]
        
        print('Discretizing categorical features...')
        if parallel:
            out = Parallel(n_jobs=n_jobs, verbose=10)( # Need to share global variables?
                delayed(smart_dummify_impute)(col_data) for col_data in ts_mixed_cols
            )
        else:
            out = [smart_dummify_impute(col_data) for col_data in tqdm(ts_mixed_cols)]
    
    out = [ts_feature_mask, *out]
    D_all = sum(len(df_i.columns) for df_i in out)
    X_all_feature_names = np.asarray(sum([list(df_i.columns) for df_i in out], []))
    X_dense = np.concatenate([df_i.values for df_i in out], axis=1).astype(dtype)
    X_all = sparse.COO(X_dense)
    
    print('Finished discretizing features')
    assert X_all.shape[0] == N * L
    X_all = X_all.reshape((N, L, D_all))
    
    print()
    print('Output')
    print('X_all: shape={}, density={:.3f}'.format(X_all.shape, X_all.density))
    return X_all, X_all_feature_names

def post_filter_time_series(X_all, feature_names_all, threshold, args):
    N, L = args.N, args.L
    assert X_all.shape[0] == N
    assert X_all.shape[1] == L
#     assert X_all.dtype == int
    start_time = time.time()
    
    X0 = X_all
    feature_names_0 = feature_names_all
    print('Original :', len(feature_names_0))
    
    ## Remove nearly-constant features (with low variance)
    sel_const = FrequencyThreshold_temporal(threshold=threshold, L=L)
    sel_const.fit(X0.reshape((N*L, -1)))
    m_ts_const = sel_const.get_support()
    assert len(m_ts_const) == X0.shape[-1]
    X1 = X0[:, :, m_ts_const]
    feature_names_1 = feature_names_0[m_ts_const]
    print('Nearly-constant:', len(feature_names_0) - len(feature_names_1))
    print('*** time: ', time.time() - start_time)
    
    ## Keep only first of pairwise perfectly correlated features
    sel_ts_corr = CorrelationSelector()
    sel_ts_corr.fit(X1.reshape((N*L, -1)))
    m_ts_corr = sel_ts_corr.get_support()
    assert len(m_ts_corr) == X1.shape[-1]
    X2 = X1[:, :, m_ts_corr]
    feature_names_2 = feature_names_1[m_ts_corr]
    feature_aliases = sel_ts_corr.get_feature_aliases(feature_names_1)
    print('Correlated     :', len(feature_names_1) - len(feature_names_2))
    print('*** time: ', time.time() - start_time)
    
    X = sparse.COO(X2)
    feature_names = feature_names_2
    assert X.shape == (N, L, len(feature_names))
    
    ## Save output
    print()
    print('Output')
    print('X: shape={}, density={:.3f}'.format(X.shape, X.density))
    
    return X, feature_names, feature_aliases
