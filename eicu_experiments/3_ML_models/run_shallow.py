# python run_shallow.py --outcome=ARF --T=4 --dt=0.5 --model_type=LR

import sys, os, time, pickle, random
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

data_path = config['data_path']
search_budget = config['train']['budget']
n_jobs = 12

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--outcome', type=str, required=True)
parser.add_argument('--T', type=float, required=True)
# parser.add_argument('--dt', type=float, required=True)
parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('--cuda', type=int, default=7)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

task = args.outcome
model_type = args.model_type
model_name = model_type

T = int(args.T)
dt = None

if model_type == 'CNN':
    assert False
elif model_type == 'RNN':
    assert False
elif model_type == 'LR':
    pass
elif model_type == 'RF':
    pass
else:
    assert False


print('EXPERIMENT:', 'model={},outcome={},T={},dt={}'.format(model_name, task, T, dt))

# Create checkpoint directories
import pathlib
pathlib.Path("./checkpoint/model={},outcome={},T={},dt={}/".format(model_name, task, T, dt)).mkdir(parents=True, exist_ok=True)
import pathlib
pathlib.Path('./output/outcome={},T={},dt={}/'.format(task, T, dt)).mkdir(parents=True, exist_ok=True)

######
# Data
import lib.data as data
tr_loader, va_loader, te_loader = data.get_train_val_test(task, duration=T, timestep=dt, fuse=True)

# Reshape feature vectors
X_tr, s_tr, y_tr = tr_loader.dataset.X, tr_loader.dataset.s, tr_loader.dataset.y
X_va, s_va, y_va = va_loader.dataset.X, va_loader.dataset.s, va_loader.dataset.y
X_te, s_te, y_te = te_loader.dataset.X, te_loader.dataset.s, te_loader.dataset.y

X_tr.shape, s_tr.shape, y_tr.shape, X_va.shape, s_va.shape, y_va.shape, X_te.shape, s_te.shape, y_te.shape, 

# Concatenate tr+va to create large training set (used for cross-validation)
Xtr = np.concatenate([X_tr, X_va])
ytr = np.concatenate([y_tr, y_va]).ravel()
Str = np.concatenate([s_tr, s_va])

Xte = X_te
yte = y_te.ravel()
Ste = s_te

# Flatten time series features
Xtr = Xtr.reshape(Xtr.shape[0], -1)
Xte = Xte.reshape(Xte.shape[0], -1)

# Combine time-invariant and time series
Xtr = np.concatenate([Xtr, Str], axis=1)
Xte = np.concatenate([Xte, Ste], axis=1)

print(Xtr.shape, ytr.shape, Xte.shape, yte.shape)
del tr_loader; del va_loader; del te_loader;


####
# sparsify
from scipy.sparse import csr_matrix
Xtr = csr_matrix(Xtr)
Xte = csr_matrix(Xte)
####


######
# Train model with CV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn import metrics, feature_selection, utils
import scipy.stats
from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm

if args.seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

if model_type == 'LR':
    clf = RandomizedSearchCV(
        LogisticRegression(solver='lbfgs'), 
        {'C': scipy.stats.reciprocal(1e-5, 1e5)},
        n_iter=search_budget,
        cv=StratifiedKFold(5),
        scoring='roc_auc',
        n_jobs=n_jobs, verbose=2,
    )
elif model_type == 'RF':
    clf = RandomizedSearchCV(
        RandomForestClassifier(), 
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [4, 8, 16, 32, None],
            "max_features": scipy.stats.randint(1, 100),
            "min_samples_split": scipy.stats.randint(2, 11),
            "min_samples_leaf": scipy.stats.randint(1, 11),
            "n_estimators": scipy.stats.randint(50,500),
            "bootstrap": [True],
        },
        n_iter=search_budget,
        cv=StratifiedKFold(5),
        scoring='roc_auc',
        n_jobs=n_jobs, verbose=2,
    )
else:
    assert False

clf.fit(Xtr, ytr)
print('best_params_', clf.best_params_)
print('best_score_ ', clf.best_score_)
try:
    np.savetxt(
        'output/outcome={},T={},dt={}/{},coef.txt'.format(task, T, dt, model_name), 
        clf.best_estimator_.coef_,
        delimiter=',',
    )
except:
    print('Coefficients not saved')
    pass

####
try:
    # save model
    import joblib
    joblib.dump(clf, "./checkpoint/model={},outcome={},T={},dt={}/".format(model_name, task, T, dt) + 'clf.joblib')
except:
    print('Model not saved')
####


###### 
# Eval
# Bootstrapped 95% Confidence Interval
try:
    yte_pred = clf.predict_proba(Xte)[:,1]
except AttributeError:
    print('Cannot produce probabilistic estimates')
    raise

def func(i):
    yte_true_b, yte_pred_b = utils.resample(yte, yte_pred, replace=True, random_state=i)
    return metrics.roc_auc_score(yte_true_b, yte_pred_b)

test_scores = Parallel(n_jobs=16)(delayed(func)(i) for i in tqdm(range(1000), leave=False))
print('Test AUC: {:.3f} ({:.3f}, {:.3f})'.format(np.median(test_scores), np.percentile(test_scores, 2.5), np.percentile(test_scores, 97.5)))

import lib.evaluate as evaluate
evaluate.save_test_predictions(y_te, yte_pred, task, T, dt, model_type)
