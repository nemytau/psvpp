from alns.data_generator import *
from alns.utils import io
from alns.Beans.schedule import Schedule
import time
# your code here
def main():
    dataset_name = 'small_1'
    base_name = 'FMO'
    insts = generate_installation_dataset(dataset_name)
    io.dump_installation_dataset(insts, dataset_name)
    base = generate_base(base_name)
    io.dump_base(base, base_name)
    fleet = generate_fleet_dataset(dataset_name)
    io.dump_fleet_dataset(fleet, dataset_name)
    # fleet = io.load_fleet_dataset(dataset_name)
    # insts = io.load_installation_dataset(dataset_name)
    # base = io.load_base(base_name)
    start = time.process_time()

    schedule = Schedule(fleet.pool, insts, base)

    print(time.process_time() - start)

if __name__ == '__main__':
    main()
