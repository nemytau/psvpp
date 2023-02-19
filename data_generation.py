from alns.data_generator import *
from alns.coord import Coord


def main():
    insts = generate_installation_dataset('small_1')
    fleet = generate_fleet_dataset()
    base = generate_base('FMO')

import numpy as np
if __name__ == '__main__':
    main()
