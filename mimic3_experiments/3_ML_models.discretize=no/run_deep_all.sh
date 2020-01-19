#!/bin/bash
set -euxo pipefail
mkdir -p log
cuda=0

python run_deep.py --outcome=mortality   --T=48.0 --dt=1.0 --model_type=CNN --cuda=$cuda &> 'log/outcome=mortality,T=48,dt=1.0,CNN.log'
python run_deep.py --outcome=mortality   --T=48.0 --dt=1.0 --model_type=RNN --cuda=$cuda &> 'log/outcome=mortality,T=48,dt=1.0,RNN.log'

python run_deep.py --outcome=ARF   --T=4.0  --dt=1.0 --model_type=CNN --cuda=$cuda &> 'log/outcome=ARF,T=4,dt=1.0,CNN.log'
python run_deep.py --outcome=ARF   --T=4.0  --dt=1.0 --model_type=RNN --cuda=$cuda &> 'log/outcome=ARF,T=4,dt=1.0,RNN.log'

python run_deep.py --outcome=ARF   --T=12.0 --dt=1.0 --model_type=CNN --cuda=$cuda &> 'log/outcome=ARF,T=12,dt=1.0,CNN.log'
python run_deep.py --outcome=ARF   --T=12.0 --dt=1.0 --model_type=RNN --cuda=$cuda &> 'log/outcome=ARF,T=12,dt=1.0,RNN.log'

python run_deep.py --outcome=Shock --T=4.0  --dt=1.0 --model_type=CNN --cuda=$cuda &> 'log/outcome=Shock,T=4,dt=1.0,CNN.log'
python run_deep.py --outcome=Shock --T=4.0  --dt=1.0 --model_type=RNN --cuda=$cuda &> 'log/outcome=Shock,T=4,dt=1.0,RNN.log'

python run_deep.py --outcome=Shock --T=12.0 --dt=1.0 --model_type=CNN --cuda=$cuda &> 'log/outcome=Shock,T=12,dt=1.0,CNN.log'
python run_deep.py --outcome=Shock --T=12.0 --dt=1.0 --model_type=RNN --cuda=$cuda &> 'log/outcome=Shock,T=12,dt=1.0,RNN.log'
