import os
import yaml
from sklearn.model_selection import ParameterGrid
import pandas as pd
import wget
import numpy as np

def load_config(filename):
    """Load a config file"""
    with open(filename) as file:
        config = yaml.load(file, Loader = yaml.FullLoader)
    return(config)

def save_config(d,filename):
    """Save a dict to a config file"""
    with open(filename, 'w') as outfile:
        yaml.dump(d, outfile, default_flow_style=False)
    
def path(pathname):
    """Create the directory if it not exist and return the absolute name of the path"""
    return_path = os.path.realpath(pathname)
    os.makedirs(return_path, exist_ok=True)
    return return_path

default_param = {'p': 36, 'std_o': 1, 'dtObs': 0.05, 'dt':0.01, 'Nfil_train':1, 'N': 20, 'seed': 10}

int_list = {'p','N','seed','Nfil_train'}


def convert(k,v):
    if k in int_list:
        return int(v)
    else:
        return v

def get_params(lparam,default_param):
    """Get the list of all the parameters (including the default ones)"""
    all_param=[]
    seq_param = ParameterGrid(lparam)
    for dparam_ in seq_param:
        all_param.append({**default_param,**dparam_})
    return all_param

def get_filenames(config, default_param):
    """return a pandas DataFrame containing filenames using the templates defined in the config files (field 'files').
    The field 'params' contains the values used to complete the filenames.
    Drop the duplicates"""
    lparam = config['params']
    files = config['files']
    seq_param = ParameterGrid(lparam)
    filenames = dict()
    def get_name(row):
        dparam = {**default_param,**dict(row)}
        dparam_convert = {k:convert(k,v) for k,v in dparam.items()}
        return files[file].format(**dparam_convert)
    for file in files:
        filenames[file] = pd.DataFrame(seq_param)
        filenames[file]['name']=filenames[file].apply(get_name, axis = 1)
        filenames[file].drop_duplicates(subset=['name'],inplace=True)
    return filenames

def load_data(indir,fname,ftpurl=None,ftpdir=None,):
    """Load data in the file fname from the the indir.
    If ftpurl and ftpdir are set, first download the data from the ftp url"""
    if ftpurl:
        assert ftpdir, 'if ftpurl is set, the argument ftpdir has to be set'
        full_url = os.path.join(ftpurl,ftpdir,fname)
        print(full_url)
        wget.download(full_url,out=indir)
    data = np.load(os.path.join(indir,fname))
    return data

def rmse_norm(pred, ref, axis=(1,2)):
    """Compute the relative RMSE between two field. By default average of the time and spatial axis"""
    norm = 2*np.var(ref)
    SE = np.square(pred-ref)/norm
    return np.sqrt(np.mean(SE,axis=axis))
