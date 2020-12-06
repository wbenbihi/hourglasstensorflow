import os
import yaml
import dacite
from utils.config import HourglassConfig

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def load_configuration(config:str):
    # Open Config Yaml File
    with open(config, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    # Parse Configuration / Data Validation
    cfg = dacite.from_dict(HourglassConfig, yaml_dict)
    return cfg