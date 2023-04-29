import pickle
from .__init__ import io_config, ROOT_PATH
import os
from enum import Enum


class DSType(Enum):
    VESSELS = 'fleet'
    INSTALLATIONS = 'installations'
    BASE = 'base'


def mkdirs():
    for dstype, path_elements in io_config['data_path'].items():
        dir_path = os.path.join(
            ROOT_PATH,
            *path_elements
        )
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def load_dataset(dataset_name, dataset_type):
    """
    Loads dataset.

    :param dataset_name: name of the dataset.
    :type dataset_name: str
    :param dataset_type: dataset type from enum DSType: VESSELS, BASE, INSTALLATIONS.
    :type dataset_type: DSType
    :return: dataset
    :rtype: list[object]|object
    """
    dstype_val = dataset_type.value
    with open(os.path.join(ROOT_PATH,
                           *io_config['data_path'][dstype_val],
                           io_config['dataset_name'][dstype_val][dataset_name]),
              'rb') as f:
        dataset = pickle.load(f)
    return dataset


def dump_dataset(dataset, dataset_name, dataset_type):
    """
        Dumps dataset.

        :param dataset: dataset object
        :param dataset_name: name of the dataset.
        :type dataset_name: str
        :param dataset_type: dataset type from enum DSType: VESSELS, BASE, INSTALLATIONS.
        :type dataset_type: DSType
        """
    dstype_val = dataset_type.value
    with open(os.path.join(ROOT_PATH,
                           *io_config['data_path'][dstype_val],
                           io_config['dataset_name'][dstype_val][dataset_name]),
              'wb') as f:
        pickle.dump(dataset, f)
