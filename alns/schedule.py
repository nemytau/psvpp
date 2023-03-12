from Beans.installation import Installation
from Beans.vessel import Vessel
from Beans.voyage import Voyage


class Schedule:

    def __init__(self, vessels: list, installations: list, schedule: list) -> None:
        self.vessels = vessels
        self.installations = installations
        if not schedule:
            self.schedule = self.generate_init_solution()
        else:
            self.schedule = schedule

    def generate_init_solution(self):
        schedule = []
        for installation in self.installations:
            scenario = installation.random_departure_scenario()
            for day in scenario:
                for vessel in self.vessels:
                    if vessel.deck_cargo +

        return schedule
