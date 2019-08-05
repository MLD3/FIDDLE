#!/bin/bash
set -euxo pipefail
mkdir -p log
mkdir -p output

python run_shallow.py --outcome=mortality --T=48.0 --dt=1.0 --model_type=LR \
    >  >(tee 'log/outcome=mortality,T=48.0,dt=1.0,LR.out') \
    2> >(tee 'log/outcome=mortality,T=48.0,dt=1.0,LR.err' >&2)
python run_shallow.py --outcome=mortality --T=48.0 --dt=1.0 --model_type=RF \
    >  >(tee 'log/outcome=mortality,T=48.0,dt=1.0,RF.out') \
    2> >(tee 'log/outcome=mortality,T=48.0,dt=1.0,RF.err' >&2)

python run_shallow.py --outcome=ARF   --T=4.0  --dt=1.0 --model_type=LR &> 'log/outcome=ARF,T=4.0,dt=1.0,LR.log'
python run_shallow.py --outcome=Shock --T=4.0  --dt=1.0 --model_type=LR &> 'log/outcome=Shock,T=4.0,dt=1.0,LR.log'

python run_shallow.py --outcome=ARF   --T=4.0  --dt=1.0 --model_type=RF &> 'log/outcome=ARF,T=4.0,dt=1.0,RF.log'
python run_shallow.py --outcome=Shock --T=4.0  --dt=1.0 --model_type=RF &> 'log/outcome=Shock,T=4.0,dt=1.0,RF.log'

python run_shallow.py --outcome=ARF   --T=12.0 --dt=1.0 --model_type=LR &> 'log/outcome=ARF,T=12.0,dt=1.0,LR.log'
python run_shallow.py --outcome=Shock --T=12.0 --dt=1.0 --model_type=LR &> 'log/outcome=Shock,T=12.0,dt=1.0,LR.log'

python run_shallow.py --outcome=ARF   --T=12.0 --dt=1.0 --model_type=RF &> 'log/outcome=ARF,T=12.0,dt=1.0,RF.log'
python run_shallow.py --outcome=Shock --T=12.0 --dt=1.0 --model_type=RF &> 'log/outcome=Shock,T=12.0,dt=1.0,RF.log'
