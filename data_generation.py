from alns.data_generator import *
from alns.utils import io
from alns.utils.io import DSType
from alns.Beans.schedule import Schedule

def main():
    io.mkdirs()
    dataset_name = 'small_1'
    base_name = 'FMO'
    insts = generate_installation_dataset(dataset_name)
    io.dump_dataset(insts, dataset_name, DSType.INSTALLATIONS)
    base = generate_base(base_name)
    io.dump_dataset(base, base_name, DSType.BASE)
    fleet = generate_fleet_dataset(dataset_name)
    io.dump_dataset(fleet, dataset_name, DSType.VESSELS)
    # fleet = io.load_dataset(dataset_name, DSType.VESSELS)
    # insts = io.load_dataset(dataset_name, DSType.INSTALLATIONS)
    # base = io.load_dataset(base_name, DSType.BASE)
    schedule = Schedule(fleet.pool, insts)

if __name__ == '__main__':
    main()
