# python run_deep.py --outcome=ARF --T=4 --dt=0.5 --model_type=CNN --cuda=7

import sys, os, time, pickle, random
import pandas as pd
import numpy as np
import pathlib
pathlib.Path('log').mkdir(parents=True, exist_ok=True)

import yaml
with open('config.yaml') as f:
    config = yaml.load(f)

########
## Constants
data_path = config['data_path']
model_names = config['model_names']

budget = config['train']['budget'] # Number of randomized hyperparameter settings to try
repeat = config['train']['repeat'] # 1   # number of restarts (with different seeds) for each setting
epochs = config['train']['epochs'] # 15  # Max epochs for each setting

# Feature dimensions
dimensions = config['feature_dimension']

# Hyperparameter search space
train_param_grid = {
    'batch_size': [16, 32, 64, 128],
    'lr': [1e-2, 1e-3, 1e-4],
}
CNN_param_grid = {
    'dropout': [0.0, 0.1, 0.2, 0.4, 0.8],
    'depth': [1, 2],#, 3],
    'filter_size': [1, 2, 3, 4],
    'n_filters': [16, 32, 64, 128],
    'n_neurons': [16, 32, 64, 128],
    'activation': ['relu', 'elu'],
}
RNN_param_grid = {
    'dropout': [0.0, 0.1, 0.2, 0.4, 0.8],
    'num_layers': [1, 2, 3],
    'hidden_size': [16, 32, 64, 128],
    'n_neurons': [16, 32, 64, 128],
    'activation': ['relu', 'elu'],
}

training_params = {'batch_size', 'lr'}

########

import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--outcome', type=str, required=True)
parser.add_argument('--T', type=float, required=True)
parser.add_argument('--dt', type=float, required=True)
parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('--cuda', type=int, default=7)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

task = args.outcome
model_type = args.model_type

T = float(args.T)
dt = float(args.dt)
L_in = int(np.floor(T / dt))
in_channels = dimensions[task][float(T)]

import lib.models as models
model_name = model_names[model_type]
ModelClass = getattr(models, model_name)

if model_type == 'CNN':
    param_grid = {**train_param_grid, **CNN_param_grid}
elif model_type == 'RNN':
    param_grid = {**train_param_grid, **RNN_param_grid}
else:
    assert False

# Create checkpoint directories
import pathlib
pathlib.Path("./checkpoint/model={}.outcome={}.T={}.dt={}/".format(model_name, task, T, dt)).mkdir(parents=True, exist_ok=True)

## Data
import lib.data as data
if task == 'mortality':
    tr_loader, va_loader, te_loader = data.get_benchmark_splits(fuse=True)
else:
    tr_loader, va_loader, te_loader = data.get_train_val_test(task, duration=T, timestep=dt, fuse=True)

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# Set CUDA 
if args.cuda:
    torch.cuda.set_device(args.cuda)
    print('cuda', torch.cuda.current_device())

if args.seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


from lib.experiment import Experiment

class MIMICExperiment(Experiment):
    def get_model_params(self, params):
        model = ModelClass(
            in_channels, L_in, 1,
            **{k:params[k] for k in params.keys() if k not in training_params}
        )
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        return model, criterion, optimizer

    def get_data(self):
        return tr_loader, va_loader

exp = MIMICExperiment(
    param_grid, name='model={}.outcome={}.T={}.dt={}'.format(model_name, task, T, dt), 
    budget=budget, n_epochs=epochs, repeat=repeat,
)

print('EXPERIMENT:', exp.name)

df_search = exp.run()
df_search.to_csv('./log/df_search.{}.csv'.format(exp.name), index=False)
