#!/bin/bash
set -euxo pipefail

export PYTHONPATH="../../"
DATAPATH=$(python -c "import yaml;print(yaml.full_load(open('../config.yaml'))['data_path']);")
mkdir -p 'log,discretize=no'

OUTCOME=ARF
T=4.0
dt=1.0
python -m FIDDLE.run \
    --data_path="$DATAPATH/features,discretize=no/outcome=$OUTCOME,T=$T,dt=$dt/" \
    --population="$DATAPATH/population/${OUTCOME}_${T}h.csv" \
    --T=$T \
    --dt=$dt \
    --theta_1=0.001 \
    --theta_2=0.001 \
    --theta_freq=1 \
    --stats_functions 'min' 'max' 'mean' \
    --discretize=no \
    > >(tee "log,discretize=no/outcome=$OUTCOME,T=$T,dt=$dt.out") \
    2> >(tee "log,discretize=no/outcome=$OUTCOME,T=$T,dt=$dt.err" >&2)

OUTCOME=ARF
T=12.0
dt=1.0
python -m FIDDLE.run \
    --data_path="$DATAPATH/features,discretize=no/outcome=$OUTCOME,T=$T,dt=$dt/" \
    --population="$DATAPATH/population/${OUTCOME}_${T}h.csv" \
    --T=$T \
    --dt=$dt \
    --theta_1=0.001 \
    --theta_2=0.001 \
    --theta_freq=1 \
    --stats_functions 'min' 'max' 'mean' \
    --discretize=no \
    > >(tee "log,discretize=no/outcome=$OUTCOME,T=$T,dt=$dt.out") \
    2> >(tee "log,discretize=no/outcome=$OUTCOME,T=$T,dt=$dt.err" >&2)

OUTCOME=Shock
T=4.0
dt=1.0
python -m FIDDLE.run \
    --data_path="$DATAPATH/features,discretize=no/outcome=$OUTCOME,T=$T,dt=$dt/" \
    --population="$DATAPATH/population/${OUTCOME}_${T}h.csv" \
    --T=$T \
    --dt=$dt \
    --theta_1=0.001 \
    --theta_2=0.001 \
    --theta_freq=1 \
    --stats_functions 'min' 'max' 'mean' \
    --discretize=no \
    > >(tee "log,discretize=no/outcome=$OUTCOME,T=$T,dt=$dt.out") \
    2> >(tee "log,discretize=no/outcome=$OUTCOME,T=$T,dt=$dt.err" >&2)

OUTCOME=Shock
T=12.0
dt=1.0
python -m FIDDLE.run \
    --data_path="$DATAPATH/features,discretize=no/outcome=$OUTCOME,T=$T,dt=$dt/" \
    --population="$DATAPATH/population/${OUTCOME}_${T}h.csv" \
    --T=$T \
    --dt=$dt \
    --theta_1=0.001 \
    --theta_2=0.001 \
    --theta_freq=1 \
    --stats_functions 'min' 'max' 'mean' \
    --discretize=no \
    > >(tee "log,discretize=no/outcome=$OUTCOME,T=$T,dt=$dt.out") \
    2> >(tee "log,discretize=no/outcome=$OUTCOME,T=$T,dt=$dt.err" >&2)



python -m FIDDLE.run \
    --data_path="$DATAPATH/features,discretize=no/benchmark,outcome=mortality,T=48.0,dt=1.0/" \
    --population="$DATAPATH/population/pop.mortality_benchmark.csv" \
    --T=48.0 \
    --dt=1.0 \
    --theta_1=0.001 \
    --theta_2=0.001 \
    --theta_freq=1 \
    --stats_functions 'min' 'max' 'mean' \
    --discretize=no \
    > >(tee 'log,discretize=no/benchmark,outcome=mortality,T=48.0,dt=1.0.out') \
    2> >(tee 'log,discretize=no/benchmark,outcome=mortality,T=48.0,dt=1.0.err' >&2)
