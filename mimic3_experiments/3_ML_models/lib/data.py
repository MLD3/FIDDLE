import sys, os, time, pickle, random
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)
    data_path = config['data_path']

def get_test(task, duration, timestep, fuse=False, batch_size=64):
    """
    Returns:
        pytorch DataLoader for test
    """
    print('Reading files')
    reader = _Mimic3Reader(task, duration, timestep)
    
    _, _, Xy_te = reader.get_splits(gap=0.0, random_state=0, verbose=False)
    te = EHRDataset(*Xy_te, fuse=fuse)
    
    num_workers = 1
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print('Time series shape, Static shape, Label shape, Class balance:')
    print('\t', 'te', te_loader.dataset.X.shape, te_loader.dataset.s.shape, te_loader.dataset.y.shape, te_loader.dataset.y.mean())
    
    if fuse:
        print('Fused dimensions:', te_loader.dataset[0][0].shape)
    
    return te_loader


def get_train_val_test(task, fuse=False, duration=4, timestep=0.5, batch_size=64):
    """
    Returns:
        pytorch DataLoader for train, val, test
    """
    print('Reading files')
    reader = _Mimic3Reader(task, duration, timestep)
    
    Xy_tr, Xy_va, Xy_te = reader.get_splits(gap=0.0, random_state=0)
    
    te = EHRDataset(*Xy_te, fuse=fuse)
    va = EHRDataset(*Xy_va, fuse=fuse)
    tr = EHRDataset(*Xy_tr, fuse=fuse)
    
    num_workers = 1
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True , num_workers=num_workers, pin_memory=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(tr_loader.dataset.y.sum() + va_loader.dataset.y.sum() + te_loader.dataset.y.sum(), '/', reader.X.shape[0])
    print('')
    print('Time series shape, Static shape, Label shape, Class balance:')
    print('\t', 'tr', tr_loader.dataset.X.shape, tr_loader.dataset.s.shape, tr_loader.dataset.y.shape, tr_loader.dataset.y.mean())
    print('\t', 'va', va_loader.dataset.X.shape, va_loader.dataset.s.shape, va_loader.dataset.y.shape, va_loader.dataset.y.mean())
    print('\t', 'te', te_loader.dataset.X.shape, te_loader.dataset.s.shape, te_loader.dataset.y.shape, te_loader.dataset.y.mean())
    
    if fuse:
        print('Fused dimensions:', tr_loader.dataset[0][0].shape)
    
    return tr_loader, va_loader, te_loader

def get_benchmark_splits(fuse=False, batch_size=64):
    task = 'mortality'
    duration = 48.0
    timestep = 1.0
    df_label = pd.read_csv(data_path + 'population/pop.mortality_benchmark.csv').rename(columns={'{}_LABEL'.format(task): 'LABEL'})
    X = sparse.load_npz(data_path +'features/benchmark,outcome={},T={},dt={}/X.npz'.format(task, duration, timestep)).todense()
    s = sparse.load_npz(data_path +'features/benchmark,outcome={},T={},dt={}/s.npz'.format(task, duration, timestep)).todense()
    
    tr_idx = df_label[df_label['partition'] == 'train'].index.values
    va_idx = df_label[df_label['partition'] == 'val'  ].index.values
    te_idx = df_label[df_label['partition'] == 'test' ].index.values
    
    def _select_examples(rows):
        return (
            X[rows], 
            s[rows], 
            df_label.iloc[rows][['LABEL']].values,
        )
    
    Xy_tr = _select_examples(tr_idx)
    Xy_va = _select_examples(va_idx)
    Xy_te = _select_examples(te_idx)
    print('ICU stay splits:', len(tr_idx), len(va_idx), len(te_idx))
    
    te = EHRDataset(*Xy_te, fuse=fuse)
    va = EHRDataset(*Xy_va, fuse=fuse)
    tr = EHRDataset(*Xy_tr, fuse=fuse)
    
    num_workers = 1
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True , num_workers=num_workers, pin_memory=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(tr_loader.dataset.y.sum() + va_loader.dataset.y.sum() + te_loader.dataset.y.sum(), '/', X.shape[0])
    print('')
    print('Time series shape, Static shape, Label shape, Class balance:')
    print('\t', 'tr', tr_loader.dataset.X.shape, tr_loader.dataset.s.shape, tr_loader.dataset.y.shape, tr_loader.dataset.y.mean())
    print('\t', 'va', va_loader.dataset.X.shape, va_loader.dataset.s.shape, va_loader.dataset.y.shape, va_loader.dataset.y.mean())
    print('\t', 'te', te_loader.dataset.X.shape, te_loader.dataset.s.shape, te_loader.dataset.y.shape, te_loader.dataset.y.mean())
    
    if fuse:
        print('Fused dimensions:', tr_loader.dataset[0][0].shape)
    
    return tr_loader, va_loader, te_loader

def get_benchmark_test(fuse=False, batch_size=64):
    task = 'mortality'
    duration = 48
    timestep = 1.0
    df_label_all = pd.read_csv(data_path + 'population/{}_{}h.csv'.format(task, duration)).rename(columns={'{}_LABEL'.format(task): 'LABEL'})
    df_label = pd.read_csv(data_path + 'population/pop.mortality_benchmark.csv').rename(columns={'{}_LABEL'.format(task): 'LABEL'})
    
    X = sparse.load_npz(data_path +'features/outcome={},T={},dt={}/X.npz'.format(task, duration, timestep)).todense()
    s = sparse.load_npz(data_path +'features/outcome={},T={},dt={}/s.npz'.format(task, duration, timestep)).todense()
    
    te_idx = [df_label_all[df_label_all['ICUSTAY_ID'] == ID].index.values[0] for ID in df_label[df_label['partition'] == 'test' ]['ID']]
    
    def _select_examples(rows):
        return (
            X[rows], 
            s[rows], 
            df_label_all.iloc[rows][['LABEL']].values,
        )
    
    Xy_te = _select_examples(te_idx)
    print('ICU stay splits:', len(te_idx))
    
    te = EHRDataset(*Xy_te, fuse=fuse)
    
    num_workers = 1
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(te_loader.dataset.y.sum())
    print('')
    print('Time series shape, Static shape, Label shape, Class balance:')
    print('\t', 'te', te_loader.dataset.X.shape, te_loader.dataset.s.shape, te_loader.dataset.y.shape, te_loader.dataset.y.mean())
    
    if fuse:
        print('Fused dimensions:', te_loader.dataset[0][0].shape)
    
    return te_loader

import sparse

class _Mimic3Reader(object):
    def __init__(self, task, duration, timestep):
        """
        """
        self.task = task
        self.duration = duration
        self.timestep = timestep
        
        start_time = time.time()
        self.df_label = pd.read_csv(data_path + 'population/{}_{}h.csv'.format(task, duration)).rename(columns={'ID': 'ICUSTAY_ID', '{}_LABEL'.format(task): 'LABEL'})
        self.df_subjects = pd.read_csv(data_path + 'prep/icustays_MV.csv').merge(self.df_label, on='ICUSTAY_ID', how='right')
        self.df_subject_label = self.df_subjects[['SUBJECT_ID', 'ICUSTAY_ID']] \
                                .merge(self.df_label, on='ICUSTAY_ID', how='right') \
                                .sort_values(by=['SUBJECT_ID', 'LABEL']) \
                                .drop_duplicates('SUBJECT_ID', keep='last').reset_index(drop=True)
        
        self.X = sparse.load_npz(data_path +'features/outcome={},T={},dt={}/X.npz'.format(task, duration, timestep)).todense()
        self.s = sparse.load_npz(data_path +'features/outcome={},T={},dt={}/s.npz'.format(task, duration, timestep)).todense()
        
        print('Finish reading data \t {:.2f} s'.format(time.time() - start_time))
    
    def _select_examples(self, rows):
        return (
            self.X[rows], 
            self.s[rows], 
            self.df_label.iloc[rows][['LABEL']].values,
        )
    
    def _select_examples_by_patients(self, subject_idx):
        subjects = self.df_subject_label.iloc[subject_idx]['SUBJECT_ID']
        stays = self.df_subjects[self.df_subjects['SUBJECT_ID'].isin(subjects)]['ICUSTAY_ID'].values
        rows = self.df_label.loc[self.df_label['ICUSTAY_ID'].isin(stays)].index.values
        return rows
    
    def _exclude_prediction_gap(self, idx, gap):
        df = self.df_label.iloc[idx]
        df = df[~(df['{}_ONSET_HOUR'.format(self.task)] < self.duration + gap)]
        return df.index.values
    
    def get_splits(self, gap=0.0, random_state=None, verbose=True):
        """
        fixed, random splits based on patient
        """
        print('Creating splits')
        tr_idx = self.df_subjects[self.df_subjects['partition'] == 'train'].index.values
        va_idx = self.df_subjects[self.df_subjects['partition'] == 'val'  ].index.values
        te_idx = self.df_subjects[self.df_subjects['partition'] == 'test' ].index.values
        try:
            import pathlib
            pathlib.Path('./output/outcome={},T={},dt={}/'.format(self.task, self.duration, self.timestep)).mkdir(parents=True, exist_ok=True)
            np.savez(open('./output/outcome={},T={},dt={}/idx.npz'.format(self.task, self.duration, self.timestep), 'wb'), tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx)
        except:
            print('indices not saved')
            raise
        
        Xy_tr = self._select_examples(tr_idx)
        Xy_va = self._select_examples(va_idx)
        Xy_te = self._select_examples(te_idx)
        print('ICU stay splits:', len(tr_idx), len(va_idx), len(te_idx))
        return Xy_tr, Xy_va, Xy_te
    
    def get_splits_stratified(self, gap=0.0, random_state=None, verbose=True):
        """
        stratified random, split based on subject
        into train (70%), val (15%), test (15%)
        """
        print('Creating splits')
        if random_state is None:
            raise UserWarning('Split results are non-deterministic unless `random_state` is set')
        
        # Create 70-15-15 stratified splits based on patient
        sss1 = StratifiedShuffleSplit(1, test_size=0.3, random_state=random_state)
        sss2 = StratifiedShuffleSplit(1, test_size=0.5, random_state=random_state)
        
        y = self.df_subject_label['LABEL'].values
        train_idx, val_test_idx = next(sss1.split(y, y))
        y_val_test = y[val_test_idx]
        val_idx, test_idx = next(sss2.split(y_val_test, y_val_test))
        val_idx = val_test_idx[val_idx]
        test_idx = val_test_idx[test_idx]
        if verbose:
            print('Patient splits:', len(train_idx), len(val_idx), len(test_idx))
        
        tr_idx = self._select_examples_by_patients(train_idx)
        va_idx = self._select_examples_by_patients(val_idx)
        te_idx = self._select_examples_by_patients(test_idx)
        if verbose:
            print('ICU stay splits:', len(tr_idx), len(va_idx), len(te_idx))
        
        tr_idx = self._exclude_prediction_gap(tr_idx, gap)
        va_idx = self._exclude_prediction_gap(va_idx, gap)
        if verbose:
            print('ICU stay splits (prediction gap):', len(tr_idx), len(va_idx), len(te_idx))
        
        try:
            import pathlib
            pathlib.Path('./output/outcome={},T={},dt={}/'.format(self.task, self.duration, self.timestep)).mkdir(parents=True, exist_ok=True)
            np.savez(open('./output/outcome={},T={},dt={}/idx.npz'.format(self.task, self.duration, self.timestep), 'wb'), tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx)
        except:
            print('indices not saved')
            raise
        
        Xy_tr = self._select_examples(tr_idx)
        Xy_va = self._select_examples(va_idx)
        Xy_te = self._select_examples(te_idx)
        
        return Xy_tr, Xy_va, Xy_te
    
    
    def get_splits_random(self, random_state=None):
        """
        70-15-15 stratified random split
        train, val, test
        """
        raise UserWarning('Not splitting by patients')
        print('Creating splits')
        if random_state is None:
            raise UserWarning('Data split is non-deterministic unless `random_state` is set')
        
        sss1 = StratifiedShuffleSplit(1, test_size=0.3, random_state=random_state)
        sss2 = StratifiedShuffleSplit(1, test_size=0.5, random_state=random_state)
        
        y = self.df_label['LABEL'].values
        train_idx, val_test_idx = next(sss1.split(y, y))
        y_val_test = y[val_test_idx]
        val_idx, test_idx = next(sss2.split(y_val_test, y_val_test))
        val_idx = val_test_idx[val_idx]
        test_idx = val_test_idx[test_idx]
        print('ICU stay splits:', len(train_idx), len(val_idx), len(test_idx))
        
        dfX_train , dfy_train = self._select_examples(train_idx)
        dfX_val,  dfy_val  = self._select_examples(val_idx)
        dfX_test, dfy_test = self._select_examples(test_idx)
        return dfX_train, dfy_train, dfX_val, dfy_val, dfX_test, dfy_test

class EHRDataset(Dataset):
    def __init__(self, X, s, y, fuse=False):
        assert len(X) == len(s)
        assert len(X) == len(y)
        self.X = X
        self.s = s
        self.y = y
        self.fuse = fuse

    def __getitem__(self, index):
        if self.fuse:
            xi = self.X[index] # LxD
            si = self.s[index] # d
            L, D = xi.shape
            xi = np.hstack((xi, np.tile(si, (L, 1))))
            return (
                torch.from_numpy(xi).float(),
                torch.from_numpy(self.y[index]).float(),
            )
        else:
            return (
                torch.from_numpy(self.X[index]).float(),
                torch.from_numpy(self.s[index]).float(),
                torch.from_numpy(self.y[index]).float(),
            )
    
    def __len__(self):
        return len(self.y)
