from .trainer import Trainer
import time
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterSampler

class Experiment(object):
    def __init__(self, param_grid, budget=1, repeat=1, n_epochs=5, name='tmp'):
        self.name = name
        self.budget = budget
        self.repeat = repeat # number of restarts with different random seeds
        self.n_epochs = n_epochs
        self.param_grid = param_grid
        self.param_sampler = ParameterSampler(param_grid, n_iter=self.budget, random_state=0)
    
    def run(self):
        df_search = pd.DataFrame(columns=['best_score', 'best_iter', 'seed', 'savename'] + list(self.param_grid.keys()))
        start_time = time.time()
        for run, params in enumerate(self.param_sampler):
            print(self.name, '\t', 'Run:', run, '/', self.budget)
            print(params)
            for i in range(self.repeat):
                results = self._run_trial(i, params)
                df_search = df_search.append(results, ignore_index=True)
                df_search.to_csv('./log/df_search.current.{}.csv'.format(self.name), index=False)

        print('Took:', time.time() - start_time)
        return df_search
    
    def _run_trial(self, seed, params):
        savename = 'checkpoint/{}/{}_seed={}.pth.tar'.format(self.name, params, seed)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        tr_loader, va_loader = self.get_data()
        model, criterion, optimizer = self.get_model_params(params)
        trainer = Trainer(model, criterion, optimizer, tr_loader, va_loader, 
                          n_epochs=self.n_epochs, batch_size=params['batch_size'], 
                          savename=savename, 
                          save_every=100, plot_every=50, cuda=True)
#         print(trainer)
        trainer.fit()

        print(trainer._best_iter, '{:.5f}'.format(trainer.best_score))
        
        del model
        return {
            'best_score': trainer.best_score, 'best_iter': trainer._best_iter, 
            'savename': savename, 'seed': seed,
            **params,
        }
    
    def get_model_params(self):
        raise NotImplementedError
    
    def get_data(self):
        raise NotImplementedError
    