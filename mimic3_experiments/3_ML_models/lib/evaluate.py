import torch
import numpy as np

def get_best_model_info(df_search):
    df_search_sorted = df_search.sort_values('best_score', ascending=False).head()
    best_model_info = df_search_sorted.iloc[0, 1:]
    return best_model_info

def load_best_model(best_model_info, ModelClass, in_channels, L_in, training_params, load_filename=None):
    if load_filename is None:
        savename = best_model_info['savename']
        split = savename.split('/')
        split[-1] = 'best_' + split[-1]
        load_filename = '/'.join(split)
    
    checkpoint = torch.load(load_filename)
    _iter = checkpoint['_iter']
    print("Loaded checkpoint (trained for {} iterations)".format(checkpoint['_iter']))
#     print(load_filename)
    
    best_HP = best_model_info.drop(['savename', 'best_iter', 'seed']).to_dict()
    model = ModelClass(
        in_channels, L_in, 1,
        **{k:best_HP[k] for k in best_HP.keys() if k not in training_params}
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    print("Restored model {} with #params={}".format(ModelClass, sum(p.numel() for p in model.parameters())))
    
    return checkpoint, model

def get_test_predictions(model, te_loader, task=None, model_name=None):
    model.eval()
    running_pred = []

    cuda = True
    for i, (X, y) in enumerate(te_loader):
        if cuda:
            X = X.contiguous().cuda()
            y = y.contiguous().cuda(non_blocking=True)

        with torch.set_grad_enabled(False):
            output = model(X)
            running_pred.append((output.data.detach().cpu(), y.data.detach().cpu()))

    y_score, y_true = zip(*running_pred)
    y_score = torch.cat(y_score).numpy()
    y_true = torch.cat(y_true).numpy()

    assert (np.stack(te_loader.dataset.y) == y_true).all()
    return y_true, y_score

def save_test_predictions(y_true, y_score, task, T, dt, model_name):
    import pathlib
    pathlib.Path('./output/outcome={}.T={}.dt={}/'.format(task, T, dt)).mkdir(parents=True, exist_ok=True)
    
    fname = './output/outcome={}.T={}.dt={}/{}.test.npz'.format(task, T, dt, model_name)
    np.savez(
        open(fname, 'wb'),
        y_score = y_score,
        y_true  = y_true,
    )
    print('Test predictions saved to', fname)
