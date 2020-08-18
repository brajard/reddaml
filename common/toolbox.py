import os
import yaml

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