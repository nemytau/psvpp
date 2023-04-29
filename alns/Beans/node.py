from itertools import combinations
import random
from alns.utils.coord import Coord
from abc import ABC


DAYS = 7


class Node(ABC):
    name: str


class Base(Node):
    def __init__(self,
                 name: str,
                 service_time: float,
                 time_window: list,
                 longitude: float,
                 latitude: float):
        self.name = name
        self.idx = 0
        self.service_time = service_time
        self.time_window = time_window
        self.longitude = longitude
        self.latitude = latitude
        self.location = Coord(longitude, latitude)

    def __repr__(self):
        return f'Base : {self.name}, location: {self.location}'


class Installation(Node):

    def __init__(self,
                 idx: int,
                 name: str,
                 inst_type: str,
                 deck_demand: int,
                 visit_frequency: int,
                 longitude: float,
                 latitude: float,
                 departure_spread: int,
                 deck_service_speed: float,
                 time_window: list) -> None:
        self.idx = idx
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

    def __hash__(self):
        return self.idx

    def __repr__(self):
        return f'{self.name}, id:{self.idx} freq:{self.visit_frequency}' \
               f' demand:{self.deck_demand} serv_time:{self.service_time}' \
               f' TW:{self.time_window}'

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
            for prev_visit_day, next_visit_day in zip(comb[:-1], comb[1:]):
                if next_visit_day - prev_visit_day <= self.departure_spread:
                    scenario_is_valid = False
                    break
            if scenario_is_valid & ((last_day + 1 + comb[0] - comb[-1]) > self.departure_spread):
                departure_scenarios.append(list(comb))
        return departure_scenarios

    def random_departure_scenario(self, period_length_days=DAYS):
        """
        Select random departure scenario.
        :param period_length_days: length of the cycled period
        :type period_length_days: int
        :return: list of departure days from randomly selected departure scenario.
        :rtype: list[int]
        """
        departure_scenarios = self._generate_departure_scenarios(list(range(period_length_days)))
        return random.choice(departure_scenarios)
