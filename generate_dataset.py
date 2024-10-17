from alns.data_generator import *
from alns.utils.utils import generate_data
from alns.utils import io


def main():
    gen_param_name = 'SMALL_1'
    dataset_name = 'test1'

    # !!! SAMPLE - for common datasets, DATA - for local
    mode = io.IOSource.SAMPLE

    generate_data(gen_param_name=gen_param_name, dataset_name=dataset_name, source=mode)


if __name__ == '__main__':
    main()