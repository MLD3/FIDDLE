from collections import defaultdict
import numpy as np
import pprint, shutil

import torch
from sklearn import metrics, utils
from datetime import datetime
from tqdm import tqdm

"""
Train the model on training data for specified number of epochs.

Evaluate on validation data and save checkpoints periodically (every several mini-batches) .

Supports running on GPU using CUDA.
"""
class Trainer(object):
    def __init__(self, model, criterion, optimizer,
                 tr_loader, va_loader, batch_size=None, n_epochs=5,
                 logdir='log/tmp/', savename='checkpoint/tmp.pth.tar',
                 save_every=100, plot_every=100,
                 cuda=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.tr_loader = _update_batch_size(tr_loader, batch_size)
        self.va_loader = _update_batch_size(va_loader, batch_size)
        
        self.n_epochs = n_epochs
        self.n_iters = 0
        
        self.plot_every = plot_every
        self.save_every = save_every
        self.savename = savename
        split = self.savename.split('/')
        split[-1] = 'best_' + split[-1]
        self.best_savename = '/'.join(split)
        
        self._epoch = 0      # number of epochs the model HAS BEEN trained
        self._iter = 0
        
        self.cuda = cuda and torch.cuda.is_available()
        if self.cuda:
            self.model = self.model.cuda()
        
        self.reset_logs()
        
    def __repr__(self):
        return pprint.pformat({
            'model': self.model,
            'optimizer': self.optimizer,
            'criterion': self.criterion,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'savename': self.savename,
        })
    
    def reset_logs(self):
        self.tr_losses = []
        self.va_losses = []
#         self.tr_scores = []
        self.va_scores = []
        self._best_iter = 0
        self.best_loss, self.best_score = self._eval(self.va_loader)

    def fit(self):
        for epoch in range(self.n_epochs):
            running_loss = []
            running_pred = []
            for i, (X, y) in tqdm(enumerate(self.tr_loader), desc='epoch '+str(epoch), leave=False):
                tr_loss = self._train_batch(X, y)
                self._iter += 1
                if self._iter % self.plot_every == 0:
                    #va_loss = self._val()
                    va_loss, va_score = self._eval(self.va_loader)
                    #tr_loss, tr_score = self._eval(self.tr_loader)
                    self.va_losses.append(va_loss)
                    self.tr_losses.append(tr_loss)
                    self.va_scores.append(va_score)
                    #self.tr_scores.append(tr_score)
#                     print(tr_loss, va_loss, va_score)
                    if self._iter % self.save_every == 0:
                        self._save(va_score)
    
    def _train_batch(self, X, y):
        self.model.train()
        if self.cuda:
            X = X.contiguous().cuda()
            y = y.contiguous().cuda(non_blocking=True)
        
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _eval(self, data_loader):
        self.model.eval()
        running_loss = []
        running_pred = []
        
        with torch.no_grad():
            for X, y in data_loader:
                if self.cuda:
                    X = X.contiguous().cuda()
                    y = y.contiguous().cuda(non_blocking=True)

                output = self.model(X)
                loss = self.criterion(output, y)
                running_loss.append(loss.item())
                running_pred.append((output.data.detach().cpu(), y.data.detach().cpu()))

        return np.mean(running_loss), self._get_score(running_pred)

    def _get_score(self, running_pred):
        y_pred, y_true = zip(*running_pred)
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        try:
            score = metrics.roc_auc_score(y_true, y_pred)
        except:
            raise
            score = metrics.accuracy_score(y_true, np.argmax(y_pred, 1))
        return score
    
    def _save(self, new_score):
        is_best = bool(new_score > self.best_score)
#         print(new_score, self.best_score, is_best)
        if is_best:
            self.best_score = new_score
            self._best_iter = self._iter
        
        state = {
            '_iter': self._iter,
            'batch_size': self.batch_size,
            'state_dict': self.model.state_dict(),
            'arch': str(type(self.model)),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'tr_losses': self.tr_losses,
            'va_losses': self.va_losses,
#             'tr_scores': self.tr_scores,
            'va_scores': self.va_scores,
        }
        
        torch.save(state, self.savename)
        if is_best:
            shutil.copyfile(self.savename, self.best_savename)

def _update_batch_size(data_loader, batch_size):
    # Alter batch size after DataLoader is created
    if batch_size:
        data_loader.batch_sampler.batch_size = batch_size
        assert np.ceil(data_loader.dataset.__len__() / batch_size) == len(iter(data_loader))
    return data_loader
