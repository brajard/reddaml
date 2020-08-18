import os
import yaml
from sklearn.model_selection import ParameterGrid
import pandas as pd

def load_config(filename):
    """Load a config file"""
    with open(filename) as file:
        config = yaml.load(file, Loader = yaml.FullLoader)
    return(config)


def path(pathname):
    """Create the directory if it not exist and return the absolute name of the path"""
    return_path = os.path.realpath(pathname)
    os.makedirs(return_path, exist_ok=True)
    return return_path

# Default value of paramter (overwrited using the configuration files)
default_param = {'p': 36, 'std_o': 1, 'dtObs': 0.05, 'dt':0.05, 'Nfil_train':1, 'N': 20, 'seed': 10}


def isint(x):
    return round(x)==x

def autoconvert(x):
    if isint(x):
        return int(x)
    else:
        return x

def get_filenames(config):
    """return a pandas DataFrame containing filenames using the templates defined in the config files (field 'files').
    The field 'params' contains the values used to complete the filenames.
    Drop the duplicates"""
    lparam = config['params']
    files = config['files']
    seq_param = ParameterGrid(lparam)
    filenames = dict()
    def get_name(row):
        dparam = {**default_param,**dict(row)}
        dparam_convert = {k:autoconvert(v) for k,v in dparam.items()}
        return files[file].format(**dparam_convert)
    for file in files:
        filenames[file] = pd.DataFrame(seq_param)
        filenames[file]['name']=filenames[file].apply(get_name, axis = 1)
        filenames[file].drop_duplicates(subset=['name'],inplace=True)
    return filenames
