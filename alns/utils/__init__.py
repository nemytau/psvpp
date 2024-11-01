import os
import yaml


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
with open(os.path.join(ROOT_PATH, 'alns', 'resource', 'io_config.yaml'), 'r') as f:
    io_config = yaml.load(f, Loader=yaml.SafeLoader)
