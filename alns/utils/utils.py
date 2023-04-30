from alns.Beans.node import Installation
from alns.utils import io
from alns.data_generator import *


def daily_visits_from_departure_scenarios(installations: list[Installation], period_length=7):
    """

    :param installations: list of installations
    :param period_length: length of the cycled period
    :return: list of installations' indices that have visit planned on the voyage starting that day
    :rtype: list[list[Installation]]
    """
    visits = [[] for n in range(0, period_length)]
    for installation in installations:
        scenario = installation.random_departure_scenario()
        for day in scenario:
            visits[day].append(installation)
    return visits


def generate_data(gen_param_name, dataset_name, base_name='FMO', source=io.IOSource.DATA):
    insts = generate_installation_dataset(gen_param_name)
    vessels = generate_vessels_dataset(gen_param_name)
    base = generate_base(base_name)

    io.dump_dataset(insts, gen_param_name, dataset_name, io.DSType.INSTALLATIONS, source=source)
    io.dump_dataset(vessels, gen_param_name, dataset_name, io.DSType.VESSELS, source=source)
    io.dump_dataset(base, gen_param_name, dataset_name, io.DSType.BASE, source=source)


def load_data(gen_param_name, dataset_name, source=io.IOSource.DATA):
    insts = io.load_dataset(gen_param_name, dataset_name, io.DSType.INSTALLATIONS, source=source)
    vessels = io.load_dataset(gen_param_name, dataset_name, io.DSType.VESSELS, source=source)
    base = io.load_dataset(gen_param_name, dataset_name, io.DSType.BASE, source=source)
    return insts, vessels, base


def load_solution(gen_param_name, dataset_name, sol_idx=0, source=io.IOSource.DATA):
    solution = io.load_dataset(gen_param_name, dataset_name, io.DSType.SOLUTION, sol_idx=sol_idx, source=source)
    return solution


def dump_solution(solution, gen_param_name, dataset_name, sol_idx=None, source=io.IOSource.DATA):
    dump_path = io.dump_dataset(solution, gen_param_name, dataset_name, io.DSType.SOLUTION, sol_idx=sol_idx, source=source)
    print(f'Solution saved at {dump_path}')
