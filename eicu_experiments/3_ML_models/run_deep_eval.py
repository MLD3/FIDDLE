import sys, os, time, pickle, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
with open('config.yaml') as f:
    config = yaml.load(f)

########
## Constants
model_names = config['model_names']
training_params = {'batch_size', 'lr'}

# Feature dimensions
dimensions = config['feature_dimension']

########

def main(task, T, dt, model_type):
    L_in = int(np.floor(T / dt))
    in_channels = dimensions[task][T]

    import lib.models as models
    model_name = model_names[model_type]
    ModelClass = getattr(models, model_name)
    df_search = pd.read_csv('./log/df_search.model={}.outcome={}.T={}.dt={}.csv'.format(model_name, task, T, dt))
    import lib.evaluate as evaluate
    best_model_info = evaluate.get_best_model_info(df_search)
    checkpoint, model = evaluate.load_best_model(best_model_info, ModelClass, in_channels, L_in, training_params)


    import lib.data as data
    if task == 'mortality':
        te_loader = data.get_benchmark_test(fuse=True)
    else:
        te_loader = data.get_test(task, duration=T, timestep=dt, fuse=True)
    
    y_true, y_score = evaluate.get_test_predictions(model, te_loader, '{}_T={}_dt={}'.format(task, T, dt), model_name)
    evaluate.save_test_predictions(y_true, y_score, task, T, dt, model_name)

    from sklearn import metrics, utils
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    fig = plt.figure(figsize=(5,5))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot([0,1], [0,1], ':')
    plt.plot(fpr, tpr, color='darkorange')
    plt.show()

    ## Bootstrapped 95% Confidence Interval
    # try:
    #     yte_pred = clf.decision_function(Xte)
    # except AttributeError:
    #     yte_pred = clf.predict_proba(Xte)[:,1]
    from sklearn.externals.joblib import Parallel, delayed
    from tqdm import tqdm_notebook as tqdm
    def func(i):
        yte_true_b, yte_pred_b = utils.resample(y_true, y_score, replace=True, random_state=i)
        return metrics.roc_auc_score(yte_true_b, yte_pred_b)

    test_scores = Parallel(n_jobs=16)(delayed(func)(i) for i in tqdm(range(1000), leave=False))
    print('Test AUC: {:.3f} ({:.3f}, {:.3f})'.format(np.median(test_scores), np.percentile(test_scores, 2.5), np.percentile(test_scores, 97.5)))

    # idx = (np.abs(tpr - 0.5)).argmin()
    # y_pred = (y_score > thresholds[idx])
    # metrics.roc_auc_score(y_true, y_score)

    precision, recall, thresholds_ = metrics.precision_recall_curve(y_true, y_score)
    fig = plt.figure(figsize=(5,5))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(recall, precision, color='darkorange')
    plt.show()

    # target TPR = 50%
    idx = (np.abs(tpr - 0.5)).argmin()
    y_pred = (y_score > thresholds[idx])
    metrics.roc_auc_score(y_true, y_score)

    pd.DataFrame([{
        'tpr': tpr[idx],
        'fpr': fpr[idx],
        'ppv': metrics.precision_score(y_true, y_pred),
    }])
