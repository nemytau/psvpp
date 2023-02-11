class Vessel:
    def __init__(self, name, deck_capacity, bulk_capacity, speed, v_type):
        self.name = name
        self.deck_capacity = deck_capacity
        self.bulk_capacity = bulk_capacity
        self.speed = speed
        self.v_type = v_type
        self.deck_cargo = 0
        self.bulk_cargo = 0

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
