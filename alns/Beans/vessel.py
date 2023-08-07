class Vessel:

    def __init__(self,
                 name: str,
                 idx: int,
                 deck_capacity: int,
                 bulk_capacity: int,
                 speed: float,
                 vessel_type: str,
                 fcs: float,
                 fcw: float,
                 cost: int,
                 ) -> None:
        self.name = name
        self.idx = idx
        self.deck_capacity = deck_capacity
        self.bulk_capacity = bulk_capacity
        self.speed = speed
        self.vessel_type = vessel_type
        self.fcs = fcs
        self.fcw = fcw
        self.cost = cost

    @classmethod
    def default_charter_vessel(cls, n):
        return cls(
            name=f'v{n}',
            deck_capacity=100,
            bulk_capacity=1000,
            speed=12,
            vessel_type='default',
            fcs=0.43,
            cost=0
        )

    def __repr__(self):
        return f'Vessel {self.name}'

    def __lt__(self, other):
        return self.deck_capacity < other.deck_capacity

    def __hash__(self):
        return self.idx
