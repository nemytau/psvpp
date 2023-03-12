from itertools import combinations
import random
from .coord import Coord


class Installation:
    def __init__(self,
                 name: str,
                 inst_type: str,
                 deck_demand: int,
                 visit_frequency: int,
                 longitude: float,
                 latitude: float,
                 departure_spread: int,
                 deck_service_speed: float,
                 time_window: list) -> None:
        self.name: str = name
        self.inst_type: str = inst_type
        self.deck_demand: int = deck_demand
        self.visit_frequency: int = visit_frequency
        self.location = Coord(longitude, latitude)
        self.departure_spread: int = departure_spread
        self.deck_service_speed: float = deck_service_speed
        self.time_window: list = time_window
        self.service_time: float = self.deck_demand / self.deck_service_speed
        # self.departure_scenarios = self._generate_departure_scenarios()
        # self.departure_days = self.random_departure_scenario()

    @classmethod
    def from_df(cls, df):
        pass

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f'{self.name} {self.inst_type}'

    def _generate_departure_scenarios(self, avail_dep_days=range(7)):
        """
        Generate departure scenarios.

        Generate departure scenarios for the installation taking into account its visit frequency and departure spread.

        :param avail_dep_days: list of numbers representing days with available departures, defaults to range(7)
        :type avail_dep_days: list[int]
        :return: list of departure scenarios
        :rtype: list[list[int]]
        """
        if self.visit_frequency * self.departure_spread > avail_dep_days[-1]:
            raise ValueError('Visit frequency and departure spread combination is infeasible for set planning horizon.')
        departure_combinations = list(combinations(avail_dep_days, self.visit_frequency))
        last_day = avail_dep_days[-1]
        departure_scenarios = []
        for comb in departure_combinations:
            scenario_is_valid = True
            for i in range(self.visit_frequency - 1):
                if comb[i + 1] - comb[i] <= self.departure_spread:
                    scenario_is_valid = False
                    break
            if scenario_is_valid & ((last_day + 1 + comb[0] - comb[-1]) > self.departure_spread):
                departure_scenarios.append(list(comb))
        return departure_scenarios

    def random_departure_scenario(self):
        """
        Select random departure scenario.

        :return: list of departure days from randomly selected departure scenario.
        :rtype: list[int]
        """
        return random.choice(self.departure_scenarios)
