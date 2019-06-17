# FIDDLE

Required packages:
- numpy
- pandas
- sparse
- sklearn
- tqdm
- joblib


Example usage:
```bash
python -m FIDDLE.run \
    --data_path='./test/small_test/' \
    --population='./test/small_test/pop.csv' \
    --T=24 --dt=5 \
    --theta_1=0.001 --theta_2=0.001 --theta_freq=1 \
    --stats_functions 'min' 'max' 'mean'
```


The generated features and associated metadata are located in `{data_path}/`:

- `s.npz`: a sparse array of shape (N, d)
- `X.npz`: a sparse tensor of shape (N, L, D)
- `s.feature_names.txt`: names of _d_ time-invariant features
- `X.feature_names.txt`: names of _D_ time-series features


To load the generated features:
```python
X = sparse.load_npz('{data_path}/X.npz'.format(data_path=...)).todense()
s = sparse.load_npz('{data_path}/s.npz'.format(data_path=...)).todense()
```
