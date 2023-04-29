class Vessel:
    def __init__(self,
                 name: int,
                 deck_capacity: int,
                 bulk_capacity: int,
                 speed: float,
                 vessel_type: str,
                 fuel_consumption: float,
                 cost: int) -> None:
        self.name = name
        self.deck_capacity = deck_capacity
        self.bulk_capacity = bulk_capacity
        self.speed = speed
        self.vessel_type = vessel_type
        self.deck_cargo = 0
        self.bulk_cargo = 0
        self.fuel_consumption = fuel_consumption
        self.cost = cost

    @classmethod
    def default_charter_vessel(cls, n):
        return cls(
            name=f'v{n}',
            deck_capacity=100,
            bulk_capacity=1000,
            speed=12,
            vessel_type='default',
            fuel_consumption=0.43,
            cost=0
        )

    def __repr__(self):
        return f'Vessel {self.name} ' \
               f'deck: {self.deck_cargo}/{self.deck_capacity}'

    def add_deck_cargo(self, amount):
        if self.deck_cargo + amount > self.deck_capacity:
            return False
        else:
            self.deck_cargo += amount

    def add_bulk_cargo(self, amount):
        if self.bulk_cargo + amount > self.bulk_capacity:
            return False
        else:
            self.bulk_cargo += amount

    def remove_deck_cargo(self, amount):
        if self.deck_cargo - amount > 0:
            return False
        else:
            self.deck_cargo -= amount

    def remove_bulk_cargo(self, amount):
        if self.bulk_cargo - amount > 0:
            return False
        else:
            self.bulk_cargo -= amount
