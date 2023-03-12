import pickle
from alns import io_config, ROOT_PATH
import os


def load_installation_dataset(dataset_name):
    with open(os.path.join(ROOT_PATH,
                           *io_config['data_path']['installations'],
                           io_config['dataset_name']['installations'][dataset_name]),
              'rb') as f:
        inst_dataset = pickle.load(f)
    return inst_dataset


def dump_installation_dataset(inst_dataset, dataset_name):
    with open(os.path.join(ROOT_PATH,
                           *io_config['data_path']['installations'],
                           io_config['dataset_name']['installations'][dataset_name]),
              'wb') as f:
        pickle.dump(inst_dataset, f)


def load_base(base_name):
    with open(os.path.join(ROOT_PATH,
                           *io_config['data_path']['base'],
                           io_config['dataset_name']['base'][base_name]),
              'rb') as f:
        base = pickle.load(f)
    return base


def dump_base(base, base_name):
    with open(os.path.join(ROOT_PATH,
                           *io_config['data_path']['base'],
                           io_config['dataset_name']['base'][base_name]),
              'wb') as f:
        pickle.dump(base, f)


def load_fleet_dataset(dataset_name):
    with open(os.path.join(ROOT_PATH,
                           *io_config['data_path']['fleet'],
                           io_config['dataset_name']['fleet'][dataset_name]),
              'rb') as f:
        fleet = pickle.load(f)
    return fleet


def dump_fleet_dataset(vessel_dataset, dataset_name):
    with open(os.path.join(ROOT_PATH,
                           *io_config['data_path']['fleet'],
                           io_config['dataset_name']['fleet'][dataset_name]),
              'wb') as f:
        pickle.dump(vessel_dataset, f)


def read_solution(solution_name):
    pass


def write_solution(solution_name):
    pass
