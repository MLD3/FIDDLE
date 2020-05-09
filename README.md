# FIDDLE

FIDDLE – <b>F</b>lex<b>I</b>ble <b>D</b>ata-<b>D</b>riven pipe<b>L</b>in<b>E</b> – is a preprocessing pipeline that transforms structured EHR data into feature vectors that can be used with ML algorithms, relying on only a small number of user-defined arguments. 

Requires python 3.6 or above. Required packages and versions are listed in `requirements.txt`. Older versions may still work but have not been tested. 

Note: This README contains latex equations and is best viewed on GitLab. 

## Publications & Resources

- Michael W. Sjoding, Shengpu Tang, Parmida Davarmanesh, Yanmeng Song, Danai Koutra, and Jenna Wiens. <b>[Democratizing EHR Analyses - a Comprehensive, Generalizable Pipeline for Learning from Clinical Data](https://www.mlforhc.org/s/Sjoding-jete.pdf)</b>. Presented at MLHC <i>([Machine Learning for Healthcare](https://www.mlforhc.org/), Clinical Abstract)</i>, 2019.
- [Poster](https://umich.box.com/s/c6rqkpd2t7gdagbjn0l5cuhaobq5zfoo)
- [MiCHAMP talk](https://umich.box.com/s/6jsrspsuj1hqldkpohv3fyz73p4902po)
- Our journal paper is currently under review by JAMIA and will be made available soon

## Usage Notes
FIDDLE generates feature vectors based on data within the observation period $`t\in[0,T]`$. This feature representation can be used to make predictions of adverse outcomes at t=T. More specifically, FIDDLE outputs a set of binary feature vectors for each example $`i`$, $`\{(s_i,x_i)\ \text{for}\ i=1 \dots N\}`$ where $`s_i \in R^d`$ contains time-invariant features and $`x_i \in R^{L \times D}`$ contains time-dependent features.

Input: 
- formatted EHR data, `.csv` or `.p`/`.pickle` files, table with 4 columns: \[`ID`, `t`, `variable_name`, `variable_value`\]
- population file: a list of unique `ID`s you want processed
- arguments:
    - T: The time of prediction; time-dependent features will be generated using data in $`t\in[0,T]`$. 
    - dt: the temporal granularity at which to "window" time-dependent data. 
    - theta_1: The threshold for Pre-filter.
    - theta_2: The threshold for Post-filter.
    - theta_freq: The threshold at which we deem a variable “frequent” (for which summary statistics will be calculated).
    - stats_functions: A set of 𝐾 statistics functions (e.g., min, max, mean). Each function is used to calculate a summary value using all recordings within a single time bin. These functions are only applicable to “frequent” variables as determined by theta_freq.

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

## Guidelines on argument settings
The user-defined arguments of FIDDLE include: T, dt, theta_1, theta_2, theta_freq, and K statistics functions. The settings of these arguments could affect the features and how they can be used. We provided reasonable default values in the implementation, and here list some practical considerations: (i) prediction time and frequency, (ii) temporal density of data, and (iii) class balance.

(i) The prediction time and frequency determine the appropriate settings for T and dt. The risk stratification tasks we considered all involve a single prediction at the end of a fixed prediction window. It is thus most reasonable to set T to be the length of prediction window. Another possible formulation is to make multiple predictions where each prediction depends on only data from the past (not the future), using models like LSTM or fully convolutional networks. In that case, for example, if a prediction needs to be made every 4 hours over a 48-hour period, then T should be 48 hours, whereas dt should be at most 4 hours. 

(ii) The temporal density of data, that is, how often the variables are usually measured, also affects the setting of dt. This can be achieved by plotting a histogram of recording frequency. In our case, we observed that the maximum hourly frequency is ~1.2 times, which suggests dt should not be smaller than 1 hour. While most variables are recorded on average <0.1 time per hour (most of the time not recorded), the 6 vital signs are recorded slightly >1 time per hour. Thus, given that in the ICU, vital signs are usually collected once per hour, we set dt=1. This also implies the setting of θ_freq to be 1. Besides determining the value for dt from context (how granular we want to encode the data), we can also sweep the range (if there are sufficient computational resources and time) given the prediction frequency and the temporal density of data. 

(iii) We recommend setting θ_1=θ_2=θ and be conservative to avoid removing information that could be potentially useful. For binary classification, the rule-of-the-thumb we suggest is to set θ to be about 1/100 of the minority class. For example, our cohorts consist of ~10% positive cases, so setting θ=0.001 is appropriate, whereas for a cohort with only 1% positive cases, then θ=0.0001 is more appropriate. Given sufficient computational resources and time, the value of θ can also be swept and optimized. 

Finally, for the summary statistics functions, we included by default the most basic statistics functions are minimum, maximum, and mean. If on average, we expect more than one value per time bin, then we can also include higher order statistics such as standard deviation and linear slope.



## Experiments

In order to show the flexibility and utility of FIDDLE, we conducted several experiments using data from MIMIC-III and eICU. The code to reproduce the results are located at https://gitlab.eecs.umich.edu/MLD3/FIDDLE_experiments. The experiments were performed using FIDDLE v0.1.0 and reported in the JAMIA paper; bug fixes and new functionalities have since been implemented and may affect the numerical results. 
