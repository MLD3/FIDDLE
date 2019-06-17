from .config import *
import pickle
import pandas as pd
import numpy as np
import time
import os

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path',       type=str,     required=True)
parser.add_argument('--population',      type=str,     required=True)
parser.add_argument('--T',               type=float,   required=True)
parser.add_argument('--dt',              type=float,   required=True)
parser.add_argument('--theta_1',         type=float,   default=0.001)
parser.add_argument('--theta_2',         type=float,   default=0.001)
parser.add_argument('--theta_freq',      type=float,   default=1.0)
parser.add_argument('--stats_functions', nargs='+',    default=['min', 'max', 'mean'])
args = parser.parse_args()

data_path = args.data_path
if not data_path.endswith('/'):
    data_path += '/'

population = args.population
T = int(args.T)
dt = args.dt
theta_1 = args.theta_1
theta_2 = args.theta_2
theta_freq = args.theta_freq
stats_functions = args.stats_functions

df_population = pd.read_csv(population).set_index('ID')
N = len(df_population)
L = int(np.floor(T/dt))

args.df_population = df_population
args.N = N
args.L = L
args.parallel = parallel

if os.path.isfile(data_path + 'input_data.p'):
    input_fname = data_path + 'input_data.p'
    df_data = pd.read_pickle(input_fname)
elif os.path.isfile(data_path + 'input_data.pickle'):
    input_fname = data_path + 'input_data.pickle'
    df_data = pd.read_pickle(input_fname)
elif os.path.isfile(data_path + 'input_data.csv'):
    input_fname = data_path + 'input_data.csv'
    df_data = pd.read_csv(input_fname)


from .steps import *

print('Input data file:', input_fname)
print()
print('Input arguments:')
print('    {:<6} = {}'.format('T', T))
print('    {:<6} = {}'.format('dt', dt))
print('    {:<6} = {}'.format('\u03B8\u2081', theta_1))
print('    {:<6} = {}'.format('\u03B8\u2082', theta_2))
print('    {:<6} = {}'.format('\u03B8_freq', theta_freq))
print('    {:<6} = {} {}'.format('k', len(stats_functions), stats_functions))
print()
print('N = {}'.format(N))
print('L = {}'.format(L))
print('', flush=True)

print_header('1) Pre-filter')
df_data = pre_filter(df_data, theta_1, df_population, args)

print_header('2) Transform; 3) Post-filter')
df_data, df_types = detect_variable_data_type(df_data, value_type_override, args)
df_time_invariant, df_time_series = split_by_timestamp_type(df_data)

# Process time-invariant data
s, s_feature_names, s_feature_aliases = transform_time_invariant(df_time_invariant, args)

# Process time-dependent data
X, X_feature_names, X_feature_aliases = transform_time_dependent(df_time_series, args)
