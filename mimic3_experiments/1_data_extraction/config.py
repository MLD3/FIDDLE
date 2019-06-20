import os, yaml
with open(os.path.join(os.path.dirname(__file__), '../config.yaml')) as f:
    config = yaml.full_load(f)

data_path = os.path.join(os.path.dirname(__file__), config['data_path'])
mimic3_path = os.path.join(os.path.dirname(__file__), config['mimic3_path'])

parallel = True
n_jobs = 72
