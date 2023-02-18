from alns.data_generator import *


def main():
    base, insts = generate_installation_dataset()
    fleet = generate_fleet_dataset()


if __name__ == '__main__':
    main()