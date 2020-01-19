#!/bin/bash
set -euxo pipefail

DATAPATH="/scratch/wiensj_root/wiensj/shared_data/FIDDLE_project/"
export PYTHONPATH="/scratch/wiensj_root/wiensj/shared_data/FIDDLE_project/"
mkdir -p log

OUTCOME=ARF
T=240.0
dt=60.0
Th=4.0
python -m FIDDLE.run \
    --data_path="$DATAPATH/features/${OUTCOME}_${Th}h/" \
    --population="$DATAPATH/population/${OUTCOME}_${Th}h.csv" \
    --T=$T \
    --dt=$dt \
    --theta_1=0.001 \
    --theta_2=0.001 \
    --theta_freq=1 \
    --stats_functions 'min' 'max' 'mean' \
    > >(tee "log/outcome=$OUTCOME,T=$T,dt=$dt.out") \
    2> >(tee "log/outcome=$OUTCOME,T=$T,dt=$dt.err" >&2)


# OUTCOME=mortality
# T=48
# dt=1.0
# python -m FIDDLE.run \
#     --data_path="$DATAPATH/features/mortality/" \
#     --population="$DATAPATH/population/${OUTCOME}_${T}h.csv" \
#     --T=$T \
#     --dt=$dt \
#     --theta_1=0.001 \
#     --theta_2=0.001 \
#     --theta_freq=1 \
#     --stats_functions 'min' 'max' 'mean' \
#     > >(tee "log/outcome=$OUTCOME,T=$T,dt=$dt.out") \
#     2> >(tee "log/outcome=$OUTCOME,T=$T,dt=$dt.err" >&2)
