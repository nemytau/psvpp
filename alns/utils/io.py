import pickle
import os
import pandas as pd
import csv
from enum import Enum
from glob import glob
from .__init__ import io_config, ROOT_PATH


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


def load_dataset(gen_param_name, dataset_name, dataset_type, sol_idx=None, source=IOSource.DATA, format='pickle'):
    """
    Loads dataset from Pickle or CSV.

    :param gen_param_name:
    :param sol_idx:
    :param source: source of loading data
    :type source: IOSource
    :param dataset_name: name of the dataset.
    :type dataset_name: str
    :param dataset_type: dataset type from enum DSType: VESSELS, BASE, INSTALLATIONS.
    :type dataset_type: DSType
    :param format: The format of the dataset file ('pickle' or 'csv').
    :type format: str
    :return: dataset
    :rtype: list[object] | object
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

    # Adjust file extension based on format
    if format == 'csv':
        filename = filename.replace('.pkl', '.csv')

    filepath = os.path.join(folderpath, filename)

    if format == 'pickle':
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
    elif format == 'csv':
        dataset = pd.read_csv(filepath).to_dict(orient='records')  # Convert to list of dictionaries (objects)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'csv'.")

    return dataset


def dump_dataset(dataset, gen_param_name, dataset_name, dataset_type, sol_idx=None, source=IOSource.DATA, format='pickle'):
    """
    Dumps dataset to Pickle or CSV.

    :param sol_idx:
    :param gen_param_name:
    :param source:
    :param dataset: dataset object
    :param dataset_name: name of the dataset.
    :type dataset_name: str
    :param dataset_type: dataset type from enum DSType: VESSELS, BASE, INSTALLATIONS.
    :type dataset_type: DSType
    :param format: The format to save the dataset ('pickle' or 'csv').
    :type format: str
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

    # Adjust file extension based on format
    if format == 'csv':
        filename = filename.replace('.pkl', '.csv')

    filepath = os.path.join(folderpath, filename)

    # Save the dataset based on the specified format
    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
    elif format == 'csv':
        if isinstance(dataset, pd.DataFrame):
            dataset.to_csv(filepath, index=False)
        elif isinstance(dataset, list) and len(dataset) > 0:
            # Convert list of objects (dictionaries) to CSV
            keys = dataset[0].__dict__.keys() if hasattr(dataset[0], '__dict__') else dataset[0].keys()
            with open(filepath, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows([obj.__dict__ if hasattr(obj, '__dict__') else obj for obj in dataset])
        else:
            raise ValueError("Dataset is neither a pandas DataFrame nor a list of objects that can be converted to CSV.")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'csv'.")

    return filename
