import os, yaml
with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
    config = yaml.full_load(f)

ID_col = config['column_names']['ID']
var_col = config['column_names']['var_name']
val_col = config['column_names']['var_value']
t_col = config['column_names']['t']

use_ordinal_encoding = config['use_ordinal_encoding']
value_type_override = config['value_types']

parallel = True
n_jobs = 72
