# FIDDLE

FIDDLE – <b>F</b>lex<b>I</b>ble <b>D</b>ata-<b>D</b>riven pipe<b>L</b>in<b>E</b> – is a preprocessing pipeline that transforms structured EHR data into feature vectors that can be used with ML algorithms, relying on only a small number of user-defined arguments.

Try a quick demo here: [tiny.cc/FIDDLE-demo](https://tiny.cc/FIDDLE-demo)

Contributions and feedback are welcome; please submit issues on the GitHub site: https://github.com/shengpu1126/FIDDLE/issues. 

## Publications & Resources
- Title: <b>Democratizing EHR analyses with FIDDLE: a flexible data-driven preprocessing pipeline for structured clinical data.</b>
- Authors: Shengpu Tang, Parmida Davarmanesh, Yanmeng Song, Danai Koutra, Michael W. Sjoding, and Jenna Wiens.
- Published in JAMIA (Journal of the American Medical Informatics Association), October 2020: [article link](https://doi.org/10.1093/jamia/ocaa139)
- Previously presented at MLHC 2019 (<i>[Machine Learning for Healthcare](https://www.mlforhc.org/)</i>) as a [clinical abstract](https://www.mlforhc.org/s/Sjoding-jete.pdf)
- News coverage on HealthcareITNews: [link](https://www.healthcareitnews.com/news/new-framework-helps-streamline-ehr-data-extraction)
- [Poster](https://www.dropbox.com/s/5rid9x12w6f8u50/MLHC%202019%20-%20FIDDLE%20poster.pdf?dl=0) | [Slides](https://www.dropbox.com/s/e6e1tfen2ae85hn/FIDDLE%20-%20MiCHAMP%2020200110%20final.pptx?dl=0)

If you use FIDDLE in your research, please cite the following publication:

```
@article{FIDDLE,
    author = {Tang, Shengpu and Davarmanesh, Parmida and Song, Yanmeng and Koutra, Danai and Sjoding, Michael W and Wiens, Jenna},
    title = "{Democratizing EHR analyses with FIDDLE: a flexible data-driven preprocessing pipeline for structured clinical data}",
    journal = {Journal of the American Medical Informatics Association},
    year = {2020},
    month = {10},
    doi = {10.1093/jamia/ocaa139},
}
```

## System Requirements

### Pip
Requires python 3.7 or above (older versions may still work but have not been tested). Required packages and versions are listed in `requirements.txt`. Run the following command to install the required packages. 
```bash
pip install -r requirements.txt
```

### Docker
To build the docker image, run the following command:
```bash
docker build -t fiddle-v020 .
```
Refer to the notebook `tests/small_test/Run-docker.ipynb` for an example to run FIDDLE in docker. 


## Usage Notes
FIDDLE generates feature vectors based on data within the observation period <img src="https://render.githubusercontent.com/render/math?math=t\in[0,T]">. This feature representation can be used to make predictions of adverse outcomes at t=T. More specifically, FIDDLE outputs a set of binary feature vectors for each example <img src="https://render.githubusercontent.com/render/math?math=i">, <img src="https://render.githubusercontent.com/render/math?math=\{(s_i,x_i)\ \text{for}\ i=1 \dots N\}"> where <img src="https://render.githubusercontent.com/render/math?math=s_i \in \mathbb{R}^d"> contains time-invariant features and <img src="https://render.githubusercontent.com/render/math?math=x_i \in \mathbb{R}^{L \times D}"> contains time-dependent features.

Input:
- formatted EHR data: `.csv` or `.p`/`.pickle` file, a table with 4 columns \[`ID`, `t`, `variable_name`, `variable_value`\]
- population file: a list of unique `ID`s you want processed
    - the output feature matrix will correspond to IDs in lexicographically sorted order
- config file:
    - specifies additional settings by providing a custom `config.yaml` file
    - a default config file is located at `FIDDLE/config-default.yaml`
- arguments:
    - T: The time of prediction; time-dependent features will be generated using data in <img src="https://render.githubusercontent.com/render/math?math=t\in[0,T]">.
    - dt: the temporal granularity at which to "window" time-dependent data.
    - theta_1: The threshold for Pre-filter.
    - theta_2: The threshold for Post-filter.
    - theta_freq: The threshold at which we deem a variable “frequent” (for which summary statistics will be calculated).
    - stats_functions: A set of 𝐾 statistics functions (e.g., min, max, mean). Each function is used to calculate a summary value using all recordings within a single time bin. These functions are only applicable to “frequent” variables as determined by theta_freq.

Output: The generated features and associated metadata are located in `{data_path}/`:

- `s.npz`: a sparse array of shape (N, d)
- `X.npz`: a sparse tensor of shape (N, L, D)
- `s.feature_names.json`: names of _d_ time-invariant features
- `X.feature_names.json`: names of _D_ time-series features
- `x.feature_aliases.json`: aliases of duplicated time-invariant features
- `X.feature_aliases.json`: aliases of duplicated time-series features


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

(iii) We recommend setting θ<sub>1</sub>=θ<sub>2</sub>=θ and be conservative to avoid removing information that could be potentially useful. For binary classification, the rule-of-the-thumb we suggest is to set θ to be about 1/100 of the minority class. For example, our cohorts consist of ~10% positive cases, so setting θ=0.001 is appropriate, whereas for a cohort with only 1% positive cases, then θ=0.0001 is more appropriate. Given sufficient computational resources and time, the value of θ can also be swept and optimized.

Finally, for the summary statistics functions, we included by default the most basic statistics functions are minimum, maximum, and mean. If on average, we expect more than one value per time bin, then we can also include higher order statistics such as standard deviation and linear slope.



## Experiments

In order to show the flexibility and utility of FIDDLE, we conducted several experiments using data from MIMIC-III and eICU. The code to reproduce the results are located at https://gitlab.eecs.umich.edu/MLD3/FIDDLE_experiments. The experiments were performed using FIDDLE v0.1.0 and reported in the JAMIA paper; bug fixes and new functionalities have since been implemented and may affect the numerical results.
