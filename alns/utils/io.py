import pickle
from .__init__ import io_config, ROOT_PATH
import os
from enum import Enum
from glob import glob


class DSType(Enum):
    VESSELS = 'vessels'
    INSTALLATIONS = 'installations'
    BASE = 'base'
    SOLUTION = 'solution'


class IOSource(Enum):
    DATA = 'data_path'
    SAMPLE = 'sample_path'


def mkdirs_all(source=IOSource.DATA):
    for dstype, path_elements in io_config[source.value].items():
        dir_path = os.path.join(
            ROOT_PATH,
            *path_elements
        )
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def load_dataset(gen_param_name, dataset_name, dataset_type, sol_idx=None, source=IOSource.DATA):
    """
    Loads dataset.

    :param gen_param_name:
    :param sol_idx:
    :param source: source of loading data
    :type source: IOSource
    :param dataset_name: name of the dataset.
    :type dataset_name: str
    :param dataset_type: dataset type from enum DSType: VESSELS, BASE, INSTALLATIONS.
    :type dataset_type: DSType
    :return: dataset
    :rtype: list[object]|object
    """
    dstype_val = dataset_type.value
    folderpath = os.path.join(ROOT_PATH,
                              *io_config[source.value][dstype_val],
                              gen_param_name)
    if dataset_type is DSType.SOLUTION:
        if sol_idx is None:
            raise AttributeError("Which solution to read? sol_idx is None")
        filename = io_config['dataset_name'][dstype_val]['prefix'] + dataset_name + \
                   f'_{sol_idx}' + io_config['dataset_name'][dstype_val]['suffix']
    else:
        filename = io_config['dataset_name'][dstype_val]['prefix'] + dataset_name + \
                   io_config['dataset_name'][dstype_val]['suffix']
    with open(os.path.join(folderpath, filename), 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def dump_dataset(dataset, gen_param_name, dataset_name, dataset_type, sol_idx=None, source=IOSource.DATA):
    """
        Dumps dataset.

        :param sol_idx:
        :param gen_param_name:
        :param source:
        :param dataset: dataset object
        :param dataset_name: name of the dataset.
        :type dataset_name: str
        :param dataset_type: dataset type from enum DSType: VESSELS, BASE, INSTALLATIONS.
        :type dataset_type: DSType
        """

    dstype_val = dataset_type.value
    folderpath = os.path.join(ROOT_PATH,
                              *io_config[source.value][dstype_val],
                              gen_param_name)
    os.makedirs(folderpath, exist_ok=True)
    if dataset_type is DSType.SOLUTION:
        filename_mask = io_config['dataset_name'][dstype_val]['prefix'] + dataset_name + '_*' + \
                        io_config['dataset_name'][dstype_val]['suffix']
        if sol_idx is None:
            possible_indices = set(range(100))
            used_indices = []
            suffix_len = -len(io_config['dataset_name'][dstype_val]['suffix'])
            for file in glob(f'{folderpath}/{filename_mask}'):
                idx = int(file.split('/')[-1].split('_')[-1][:suffix_len])
                used_indices.append(idx)
            sol_idx = list(possible_indices - set(used_indices))[0]
        filename = filename_mask.replace('*', str(sol_idx))
    else:
        filename = io_config['dataset_name'][dstype_val]['prefix'] + dataset_name + \
                   io_config['dataset_name'][dstype_val]['suffix']
    with open(os.path.join(folderpath, filename), 'wb') as f:
        pickle.dump(dataset, f)
    return filename
