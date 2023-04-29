from alns.data_generator import *
from alns.utils import io
from alns.alns.alns import ALNS
from config.config_utils import get_config
import time


def main():
    dataset_name = 'small_1'
    base_name = 'FMO'
    insts = generate_installation_dataset(dataset_name)
    io.dump_installation_dataset(insts, dataset_name)
    base = generate_base(base_name)
    io.dump_base(base, base_name)
    # fleet = generate_fleet_dataset(dataset_name)
    # io.dump_fleet_dataset(fleet, dataset_name)
    # fleet = io.load_fleet_dataset(dataset_name)
    # insts = io.load_installation_dataset(dataset_name)
    # base = io.load_base(base_name)

    # schedule = Schedule(fleet.pool, insts, base)
    # alns = ALNS(
    #     installations=insts,
    #     base=base,
    #     fleet=fleet.pool,
    #     iterations=int(get_config()["alns"]["iterations"]),
    #     speed_coeff=float(get_config()["alns"]["speed_up_coeff"]),
    #     operator_select_type="Stochastic")
    alns = ALNS(
            installations=insts,
            base=base,
            fleet=fleet.pool,
            iterations=100,
            speed_coeff=0.7,
            operator_select_type="Stochastic")

    # alns.start(int(get_config()["alns"]["repetitions"]))
    alns.start(10)



if __name__ == '__main__':
    main()
