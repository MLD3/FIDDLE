import os, yaml
import copy

with open(os.path.join(os.path.dirname(__file__), 'config-default.yaml')) as f:
    config_default = yaml.safe_load(f)

def load_config(fname):
    config = copy.deepcopy(config_default)
    if fname:
        config_custom = yaml.safe_load(open(fname, 'r'))
        for k, v in config_custom.items():
            config[k] = v
    return config


ID_col = 'ID'
t_col = 't'
var_col = 'variable_name'
val_col = 'variable_value'

if 'column_names' in config_default:
    ID_col = config_default['column_names'].get('ID', 'ID')
    t_col = config_default['column_names'].get('t', 't')
    var_col = config_default['column_names'].get('var_name', 'variable_name')
    val_col = config_default['column_names'].get('var_value', 'variable_value')
else:
    pass
