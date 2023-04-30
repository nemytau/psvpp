from alns.data_generator import *
from alns.utils import io
from alns.alns.alns import ALNS
from alns.Beans.schedule import Schedule
from config.config_utils import get_config
import time


def main():
    dataset_name = 'small_1'
    base_name = 'FMO'
    solution_names = [f'sol{i}_{dataset_name}' for i in range(1,6)]
    solution_name = solution_names[0]

    insts = generate_installation_dataset(dataset_name)
    vessels = generate_vessels_dataset(dataset_name)
    base = generate_base(base_name)
    #
    # io.dump_dataset(insts, dataset_name, io.DSType.INSTALLATIONS)
    # io.dump_dataset(vessels, dataset_name, io.DSType.VESSELS)
    # io.dump_dataset(base, base_name, io.DSType.BASE)

    # insts = io.load_dataset(dataset_name, io.DSType.INSTALLATIONS)
    # vessels = io.load_dataset(dataset_name, io.DSType.VESSELS)
    # base = io.load_dataset(base_name, io.DSType.BASE)

    schedule = Schedule(vessels, insts, base)
    io.dump_dataset(schedule, solution_name, io.DSType.SOLUTION)


    # alns = ALNS(
    #     installations=insts,
    #     base=base,
    #     fleet=fleet.pool,
    #     iterations=int(get_config()["alns"]["iterations"]),
    #     speed_coeff=float(get_config()["alns"]["speed_up_coeff"]),
    #     operator_select_type="Stochastic")
    #
    # alns.start(int(get_config()["alns"]["repetitions"]))


if __name__ == '__main__':
    main()
