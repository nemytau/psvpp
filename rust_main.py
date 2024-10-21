from alns.utils import io
from alns.utils.utils import load_data

def pkl_to_csv_dataset():
    gen_param_name = 'SMALL_1'
    dataset_name = 'test1'

    insts, vessels, base = load_data(gen_param_name=gen_param_name, dataset_name=dataset_name,
                                     source=io.IOSource.SAMPLE)
    io.dump_dataset(insts, gen_param_name, dataset_name, io.DSType.INSTALLATIONS, source=io.IOSource.SAMPLE, format='csv')
    io.dump_dataset(vessels, gen_param_name, dataset_name, io.DSType.VESSELS, source=io.IOSource.SAMPLE, format='csv')
    io.dump_dataset([base], gen_param_name, dataset_name, io.DSType.BASE, source=io.IOSource.SAMPLE, format='csv')


def main():
    gen_param_name = 'SMALL_1'
    dataset_name = 'test1'
    
    
if __name__ == '__main__':
    main()
