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

def get_test(task, duration, timestep=None, fuse=False, batch_size=64):
    """
    Returns:
        pytorch DataLoader for test
    """
    print('Reading files')
    reader = _eICUReader(task, duration, timestep)
    
    _, _, Xy_te = reader.get_splits(gap=0.0, random_state=0, verbose=False)
    te = EHRDataset(*Xy_te, fuse=fuse)
    
    num_workers = 1
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print('Time series shape, Static shape, Label shape, Class balance:')
    print('\t', 'te', te_loader.dataset.X.shape, te_loader.dataset.s.shape, te_loader.dataset.y.shape, te_loader.dataset.y.mean())
    
    if fuse:
        print('Fused dimensions:', te_loader.dataset[0][0].shape)
    
    return te_loader


def get_train_val_test(task, fuse=False, duration=4, timestep=None, batch_size=64):
    """
    Returns:
        pytorch DataLoader for train, val, test
    """
    print('Reading files')
    reader = _eICUReader(task, duration, timestep)
    
    Xy_tr, Xy_va, Xy_te = reader.get_splits()
    
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

import sparse

class _eICUReader(object):
    def __init__(self, task, duration, timestep=None):
        """
        """
        self.task = task
        self.duration = duration
        
        start_time = time.time()
        self.df_label = pd.read_csv(data_path + 'population/{}_{}h.csv'.format(task, duration)).rename(columns={'ID': 'ICUSTAY_ID', '{}_LABEL'.format(task): 'LABEL'})
        self.df_subjects = pd.read_csv(data_path + 'icustays.csv').rename(columns={'PatientID': 'SUBJECT_ID', 'ICUStayID': 'ICUSTAY_ID'}).merge(self.df_label, on='ICUSTAY_ID', how='right')
        self.df_subject_label = self.df_subjects[['SUBJECT_ID', 'ICUSTAY_ID']] \
                                .merge(self.df_label, on='ICUSTAY_ID', how='right') \
                                .sort_values(by=['SUBJECT_ID', 'LABEL']) \
                                .drop_duplicates('SUBJECT_ID', keep='last').reset_index(drop=True)
        
        self.X = sparse.load_npz(data_path +'features/{}_{}h_download/X.npz'.format(task, duration)).todense()
        self.s = sparse.load_npz(data_path +'features/{}_{}h_download/s.npz'.format(task, duration)).todense()
        
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
            pathlib.Path('./output/{},{}h/'.format(self.task, self.duration)).mkdir(parents=True, exist_ok=True)
            np.savez(open('./output/{},{}h/idx.npz'.format(self.task, self.duration), 'wb'), tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx)
        except:
            print('indices not saved')
            raise
        
        Xy_tr = self._select_examples(tr_idx)
        Xy_va = self._select_examples(va_idx)
        Xy_te = self._select_examples(te_idx)
        print('ICU stay splits:', len(tr_idx), len(va_idx), len(te_idx))
        return Xy_tr, Xy_va, Xy_te

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
