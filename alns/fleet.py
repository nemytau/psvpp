from .vessel import Vessel


class Fleet:
    def __init__(self) -> None:
        self.pool = []
        self.vessel = {}

    @classmethod
    def default_fleet(cls, num):
        fleet = cls()
        for i in range(num):
            fleet.add_vessel(Vessel.default_charter_vessel(i))
        return fleet

    @classmethod
    def from_vessels_list(cls, vessels):
        fleet = cls()
        for vessel in vessels:
            fleet.add_vessel(vessel)
        return fleet

    def __repr__(self):
        return '\n'.join([str(vessel) for vessel in self.pool])

    def add_vessel(self, vessel):
        self.pool.append(vessel)
        self.vessel[vessel.name] = vessel

    def remove_vessel(self, vessel):
        self.vessel.pop(vessel.name)
        self.pool.remove(vessel)

    def pool_cost(self):
        return [vessel.cost for vessel in self.pool]

    # select from enum?
    def add_empty_vessel(self):
        pass

    def find_empty_vessel(self, capacity_asc=None):
        if capacity_asc is None:
            for vessel in self.pool:
                if (vessel.bulk_cargo + vessel.deck_cargo) == 0:
                    return vessel
        if capacity_asc:
            # return first largest empty
            pass
        