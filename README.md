# FIDDLE

FIDDLE – <b>F</b>lex<b>I</b>ble <b>D</b>ata-<b>D</b>riven pipe<b>L</b>in<b>E</b> – is a preprocessing pipeline that transforms structured EHR data into feature vectors that can be used with ML algorithms, relying on only a small number of user-defined arguments. 

Requires python 3.6 or above. Required packages and versions are listed in `requirements.txt`. Older versions may still work but have not been tested. 

## Usage Notes
FIDDLE generates feature vectors based on data within the observation period $`t\in[0,T]`$. This feature representation can be used to make predictions of adverse outcomes at t=T. More specifically, FIDDLE outputs a set of binary feature vectors for each example $`i`$, $`\{(s_i,x_i)\ \text{for}\ i=1 \dots N\}`$ where $`s_i \in R^d`$ contains time-invariant features and $`x_i \in R^{L \times D}`$ contains time-dependent features.

Input: 
- formatted EHR data, `.csv` or `.p`/`.pickle` files, table with 4 columns: \[`ID`, `t`, `variable_name`, `variable_value`\]
- population file: a list of unique `ID`s you want processed
- arguments:
    - T: the prediction time. Time-dependent features will be generated using data in $`t\in[0,T]`$. 
    - dt: the temporal granularity at which to "window" time-dependent data. 
    - theta_1
    - theta_2
    - theta_freq

Output: The generated features and associated metadata are located in `{data_path}/`:

- `s.npz`: a sparse array of shape (N, d)
- `X.npz`: a sparse tensor of shape (N, L, D)
- `s.feature_names.txt`: names of _d_ time-invariant features
- `X.feature_names.txt`: names of _D_ time-series features


To load the generated features:
```python
X = sparse.load_npz('{data_path}/X.npz'.format(data_path=...)).todense()
s = sparse.load_npz('{data_path}/s.npz'.format(data_path=...)).todense()
```


Example usage:
```bash
python -m FIDDLE.run \
    --data_path='./test/small_test/' \
    --population='./test/small_test/pop.csv' \
    --T=24 --dt=5 \
    --theta_1=0.001 --theta_2=0.001 --theta_freq=1 \
    --stats_functions 'min' 'max' 'mean'
```

## Experiments

In order to show the flexibility and utility of FIDDLE, swe conducted several experiments using data from MIMIC-III. The code to reproduce the results are located in the `mimic3_experiments` subdirectory. 
