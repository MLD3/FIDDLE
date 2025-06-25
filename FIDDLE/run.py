import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import pickle
import time
import os
import yaml
import json
import argparse

import FIDDLE.config as FIDDLE_config
import FIDDLE.steps as FIDDLE_steps

def main():
    ######
    # User arguments
    ######
    parser = argparse.ArgumentParser(description='')

    # Files
    parser.add_argument('--data_fname',      type=str,     required=True)
    parser.add_argument('--population_fname',type=str,     required=True)
    parser.add_argument('--output_dir',      type=str,     required=True)
    parser.add_argument('--config_fname',    type=str,     required=False)

    # Settings
    parser.add_argument('--T',               type=float,   required=True)
    parser.add_argument('--dt',              type=float,   required=True)
    parser.add_argument('--theta_1',         type=float,   default=0.001)
    parser.add_argument('--theta_2',         type=float,   default=0.001)
    parser.add_argument('--theta_freq',      type=float,   default=1.0)
    parser.add_argument('--stats_functions', nargs='+',    default=['min', 'max', 'mean'])

    # Debug
    parser.add_argument('--N',               type=int,     required=False)
    parser.add_argument('--Ds',              nargs='+',    type=int)
    parser.add_argument('--no_prefilter',    dest='prefilter',  action='store_false')
    parser.add_argument('--no_postfilter',   dest='postfilter', action='store_false')
    parser.set_defaults(prefilter=True, postfilter=True)

    args = parser.parse_args()


    ######
    # Load files
    ######

    data_fname = args.data_fname
    if data_fname.endswith('.p' or '.pickle'):
        df_data = pd.read_pickle(data_fname)
    elif data_fname.endswith('.csv'):
        df_data = pd.read_csv(data_fname)
    else:
        raise NotImplementedError

    df_population = args.df_population = pd.read_csv(args.population_fname).set_index('ID').sort_index()
    config = FIDDLE_config.load_config(args.config_fname)


    ## Arguments settings
    output_dir = args.output_dir
    if not output_dir.endswith('/'):
        output_dir += '/'

    T = args.T
    dt = args.dt
    theta_1 = args.theta_1
    theta_2 = args.theta_2
    theta_freq = args.theta_freq
    stats_functions = args.stats_functions

    args.hierarchical_sep = config.get('hierarchical_sep', ':')
    args.hierarchical_levels = config.get('hierarchical_levels', [])
    args.value_type_override = config.get('value_types', {})

    args.discretize = config.get('discretize', True)
    args.use_ordinal_encoding = config.get('use_ordinal_encoding', False)

    args.S_discretization_bins = None
    args.X_discretization_bins = None
    S_discretization_bins = config.get('S_discretization_bins')
    X_discretization_bins = config.get('X_discretization_bins')
    if S_discretization_bins:
        args.S_discretization_bins = json.load(open(S_discretization_bins, 'r'))
    if X_discretization_bins:
        args.X_discretization_bins = json.load(open(X_discretization_bins, 'r'))

    args.parallel = config.get('parallel', False)
    args.n_jobs = config.get('n_jobs', 1)
    args.batch_size = config.get('batch_size', 100)

    N = args.N = args.N or len(df_population)
    df_population = df_population.iloc[:args.N]
    L = args.L = int(np.floor(T/dt))

    print('Input:')
    print('    Data      :', args.data_fname)
    print('    Population:', args.population_fname)
    print('    Config    :', args.config_fname)
    print()
    print('Output directory:', args.output_dir)
    print()
    print('Input arguments:')
    print('    {:<6} = {}'.format('T', T))
    print('    {:<6} = {}'.format('dt', dt))
    print('    {:<6} = {}'.format('\u03B8\u2081', theta_1))
    print('    {:<6} = {}'.format('\u03B8\u2082', theta_2))
    print('    {:<6} = {}'.format('\u03B8_freq', theta_freq))
    print('    {:<6} = {} {}'.format('k', len(stats_functions), stats_functions))
    print()
    print('{} = {}'.format('discretize', {False: 'no', True: 'yes'}[args.discretize]))
    if args.discretize:
        print('    S discretization bins:', S_discretization_bins or 'to be computed from data')
        print('    X discretization bins:', X_discretization_bins or 'to be computed from data')
    print()
    print('N = {}'.format(N))
    print('L = {}'.format(L))
    print('', flush=True)


    ######
    # Main
    ######
    df_population[[]].to_csv(output_dir + 'IDs.csv')

    if args.prefilter:
        FIDDLE_steps.print_header('1) Pre-filter')
        df_data = FIDDLE_steps.pre_filter(df_data, theta_1, df_population, args)
        df_data.to_csv(output_dir + 'pre-filtered.csv', index=False)

    FIDDLE_steps.print_header('2) Transform; 3) Post-filter')
    df_data, df_types = FIDDLE_steps.parse_variable_data_type(df_data, args)
    df_time_invariant, df_time_series = FIDDLE_steps.split_by_timestamp_type(df_data)

    # Process time-invariant data
    S, S_feature_names, S_feature_aliases = FIDDLE_steps.process_time_invariant(df_time_invariant, args)

    # Process time-dependent data
    X, X_feature_names, X_feature_aliases = FIDDLE_steps.process_time_dependent(df_time_series, args)

if __name__ == '__main__':
    main()
