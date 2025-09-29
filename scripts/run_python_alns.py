from py_alns.data_generator import *
from py_alns.utils import io
from py_alns.utils.utils import *
from py_alns.alns.alns import ALNS
from py_alns.Beans.schedule import Schedule
from config.config_utils import get_config
import time



def main():
    gen_param_name = 'SMALL_1'
    dataset_name = 'test1'

    insts, vessels, base = load_data(gen_param_name=gen_param_name, dataset_name=dataset_name,
                                     source=io.IOSource.SAMPLE)

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
