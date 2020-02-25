from .config import *
import pickle
import pandas as pd
import numpy as np
import time
import os

import argparse
from .helpers import str2bool

parser = argparse.ArgumentParser(description='')
parser.add_argument('--T',               type=float,   required=True)
parser.add_argument('--dt',              type=float,   required=True)
parser.add_argument('--theta_1',         type=float,   default=0.001)
parser.add_argument('--theta_2',         type=float,   default=0.001)
parser.add_argument('--theta_freq',      type=float,   default=1.0)
parser.add_argument('--stats_functions', nargs='+',    default=['min', 'max', 'mean'])
parser.add_argument('--binarize',        type=str2bool, default=True, nargs='?', const=True)

parser.add_argument('--data_path',       type=str,     required=True)
parser.add_argument('--input_fname',     type=str,     required=False)
parser.add_argument('--population',      type=str,     required=True)
parser.add_argument('--N',               type=int,     required=False)
parser.add_argument('--Ds',              nargs='+',    type=int)

parser.add_argument('--no_prefilter',    dest='prefilter',  action='store_false')
parser.add_argument('--no_postfilter',   dest='postfilter', action='store_false')
parser.set_defaults(prefilter=True, postfilter=True)

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
binarize = args.binarize

df_population = pd.read_csv(population).set_index('ID')
N = args.N or len(df_population)
df_population = df_population.iloc[:args.N]
L = int(np.floor(T/dt))

args.df_population = df_population
args.N = N
args.L = L
args.parallel = parallel

if args.input_fname and os.path.isfile(args.input_fname):
    input_fname = args.input_fname
    if input_fname.endswith('.p' or '.pickle'):
        df_data = pd.read_pickle(input_fname)
    elif input_fname.endswith('.csv'):
        df_data = pd.read_csv(input_fname)
    else:
        assert False
elif os.path.isfile(data_path + 'input_data.p'):
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
print('{} = {}'.format('binarize', {False: 'no', True: 'yes'}[binarize]))
print()
print('N = {}'.format(N))
print('L = {}'.format(L))
print('', flush=True)


######
# Main
######
if args.prefilter:
    print_header('1) Pre-filter')
    df_data = pre_filter(df_data, theta_1, df_population, args)
    df_data.to_csv(data_path + 'pre-filtered.csv', index=False)

print_header('2) Transform; 3) Post-filter')
df_data, df_types = detect_variable_data_type(df_data, value_type_override, args)
df_time_invariant, df_time_series = split_by_timestamp_type(df_data)

# Process time-invariant data
s, s_feature_names, s_feature_aliases = process_time_invariant(df_time_invariant, args)

# Process time-dependent data
X, X_feature_names, X_feature_aliases = process_time_dependent(df_time_series, args)
