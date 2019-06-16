## Overview
- Requires: the raw csv files of [MIMIC-III database](https://mimic.physionet.org/about/mimic/)
- Extract and format data from structured tables in MIMIC-III as input to FIDDLE
- Goal: using data from all tables, generate Time-Invariant features **s** and Time Series features __X__. 

## Usage

0. Modify `config.yaml` to specify `mimic3_path` and `data_path`.

1. Data Extraction
    - Execute `python -c "from extract_data import *; check_nrows();"` to verify the integrity of raw csv files.
    - Run `python extract_data.py`.

1. Labels & Analyses
    - Run `python generate_labels.py` to generate the event onset time and labels for two outcomes: ARF and shock.
    - Run the following notebooks in order: `LabelDistribution.ipynb`, `InclusionExclusion.ipynb` and `PopulationSummary.ipynb`.

1. Generate features
    - Run `python prepare_input.py --task={task} --T={T} --dt={dt}`
    - Run `python make_features.py --task={task} --T={T} --dt={dt}`
    
    Note: a bash script is provided for generating features.

The generated features and associated metadata are located in `{data_path}/features/task={task}.T={T}.dt={dt}/`:

- `s.npz`: a sparse array of shape (N, d)
- `X.npz`: a sparse tensor of shape (N, L, D)
- `s.feature_names.txt`: names of _d_ time-invariant features
- `X.feature_names.txt`: names of _D_ time-series features
