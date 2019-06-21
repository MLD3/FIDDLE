#!/bin/bash
set -euxo pipefail

python prepare_input.py --outcome=ARF   --T=4  --dt=1
python prepare_input.py --outcome=ARF   --T=12 --dt=1
python prepare_input.py --outcome=Shock --T=4  --dt=1
python prepare_input.py --outcome=Shock --T=12 --dt=1

python prepare_input.py --outcome=mortality --T=48 --dt=1
mv ../data/processed/features/outcome=mortality,T=48.0,dt=1.0 ../data/processed/features/benchmark,outcome=mortality,T=48.0,dt=1.0
