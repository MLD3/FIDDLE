import os, yaml
with open(os.path.join(os.path.dirname(__file__), '../config.yaml')) as f:
    config = yaml.full_load(f)

data_path = os.path.join(os.path.dirname(__file__), config['data_path'])
mimic3_path = os.path.join(os.path.dirname(__file__), config['mimic3_path'])

ID_col = config['column_names']['ID']
t_col = config['column_names']['t']
var_col = config['column_names']['var_name']
val_col = config['column_names']['var_value']

parallel = True
n_jobs = 72
