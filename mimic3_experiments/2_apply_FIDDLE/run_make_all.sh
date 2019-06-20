#!/bin/bash
set -euxo pipefail

export PYTHONPATH="$PYTHONPATH:../../"
DATAPATH=$(python -c "import yaml;print(yaml.full_load(open('../config.yaml'))['data_path']);")


python prepare_input.py --outcome=ARF --T=4  --dt='1.0'
# python prepare_input.py --outcome=ARF --T=12 --dt='1.0'
# python prepare_input.py --outcome=Shock --T=4  --dt='1.0'
# python prepare_input.py --outcome=Shock --T=12 --dt='1.0'

# python make_features.py --outcome=ARF --T=4  --dt='1.0' > 'log/outcome=ARF.T=4.dt=1.0.out' 2> 'log/outcome=ARF.T=4.dt=1.0.err'
# python make_features.py --outcome=ARF --T=12 --dt='1.0' > >(tee 'log/outcome=ARF.T=12.dt=1.0.out') 2> >(tee 'log/outcome=ARF.T=12.dt=1.0.err' >&2)
# python make_features.py --outcome=Shock --T=4  --dt='1.0' > >(tee 'log/outcome=Shock.T=4.dt=1.0.out') 2> >(tee 'log/outcome=Shock.T=4.dt=1.0.err' >&2)
# python make_features.py --outcome=Shock --T=12 --dt='1.0' > >(tee 'log/outcome=Shock.T=12.dt=1.0.out') 2> >(tee 'log/outcome=Shock.T=12.dt=1.0.err' >&2)
# python make_features.py \
#     --data_path='/data4/tangsp/mimic3_features/features/outcome=mortality.T=48.dt=1.0/' \
#     --population='/data4/tangsp/mimic3_features/population/mortality_48h.csv' \
#     --T=48 --dt='1.0' \
#     > >(tee 'log/outcome=mortality.T=48.dt=1.0.out') \
#     2> >(tee 'log/outcome=mortality.T=48.dt=1.0.err' >&2)

python make_features.py \
    --data_path='/data4/tangsp/mimic3_features/features/outcome=mortality.T=48.dt=1.0/' \
    --population='/data4/tangsp/mimic3_features/population/mortality_48h.csv' \
    --T=48 --dt='1.0'
    
python -m FIDDLE.run \
    --data_path='./small_test/' \
    --population='./small_test/pop.csv' \
    --T=4 \
    --dt=1.0 \
    --theta_1=0.001 \
    --theta_2=0.001 \
    --theta_freq=1 \
    --stats_functions 'min' 'max' 'mean' \
    > >(tee 'log/outcome=mortality.T=48.dt=1.0.out') \
    2> >(tee 'log/outcome=mortality.T=48.dt=1.0.err' >&2)
