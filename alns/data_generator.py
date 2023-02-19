from .__init__ import generation_yaml_config
import pandas as pd
import numpy as np
from .installation import Installation
from .base import Base


def generate_installation_dataframe(inst_sample_name):
    sample_config = generation_yaml_config['inst_generation_params'][inst_sample_name]
    inst_df_list = []
    total_inst_num = sum([conf['num'] for conf in sample_config])
    inst_indices = np.random.choice(a=range(1, len(generation_yaml_config['inst_coords'])),
                                    size=total_inst_num,
                                    replace=False
                                    )
    coords = np.array(generation_yaml_config['inst_coords'])
    for inst_type_config in sample_config:
        df = pd.DataFrame(index=range(inst_type_config['num']))
        df['name'] = inst_type_config['name_prefix'] + df.index.astype(str)
        df['inst_type'] = inst_type_config['type']
        df['deck_demand'] = np.round(np.random.normal(inst_type_config['demand_mu'],
                                                      inst_type_config['demand_sigma'],
                                                      inst_type_config['num'])).astype(int)
        df['visit_frequency'] = np.random.choice(inst_type_config['frequencies']['values'],
                                                 size=inst_type_config['num'],
                                                 p=inst_type_config['frequencies']['probs'])

        inst_type_indices = inst_indices[:inst_type_config['num']]
        inst_indices = inst_indices[inst_type_config['num']:]
        df['longitude'] = coords[inst_type_indices, 0]
        df['latitude'] = coords[inst_type_indices, 1]
        df['departure_spread'] = inst_type_config['departure_spread']
        df['deck_service_speed'] = inst_type_config['deck_service_speed']
        df['time_window'] = '0, 24'
        df.loc[df.sample(inst_type_config['time_windows']).index,
               'time_window'] = inst_type_config['default_time_window']
        df['time_window'] = df['time_window'].str.split(',').map(tuple)
        inst_df_list.append(df)
    return pd.concat(inst_df_list).reset_index(drop=True)


def generate_installation_dataset(inst_sample_name):
    inst_df = generate_installation_dataframe(inst_sample_name)
    insts = inst_df.apply(lambda x: Installation(**(x.to_dict())), axis=1).to_list()
    return insts


def generate_base(name):
    base_config = generation_yaml_config['base_generation_params']
    base = Base(
        name,
        base_config[name]['service_time'],
        tuple(map(int, base_config[name]['time_window'].split(',')))
    )
    return base


def installation_dataset_from_file():
    pass


def generate_fleet_dataset():
    pass


def installation_fleet_from_file():
    pass
