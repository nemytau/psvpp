from .route import Route
from alns.Beans.vessel import Vessel


class Voyage:
    DAYS = 7
    HOURS = 24
    PERIOD_LENGTH = 168

    def __init__(self,
                 vessel: Vessel,
                 route: Route,
                 start_time: float):
        self.vessel = vessel
        self.route = route
        self.start_time = start_time
        self.voyage_length = self.calc_voyage_length()
        self.end_time = (self.voyage_length + self.start_time) % self.PERIOD_LENGTH
        self.load = self.calc_load()

    def calc_load(self):
        return sum([inst.deck_demand for inst in self.route.route[1:-1]])

    def calc_voyage_length(self, ):
        """
        Calculates time length of the voyage.

        :return: time length of the voyage in hours
        :rtype: float
        """
        end_time = self.start_time
        # print(f'START {start_time}')
        for edge in self.route.edges():
            to_node = edge[1]
            dist = edge[2]
            arrival_cum_time = (end_time + dist/self.vessel.speed)
            # print(f'ARRIVED {arrival_cum_time}, SAILED {dist}, TW: {to_node.time_window}')
            arrival_day = arrival_cum_time // self.HOURS
            arrival_time = arrival_cum_time % self.HOURS
            if arrival_time < to_node.time_window[0]:
                start_service_time = arrival_day * self.HOURS + to_node.time_window[0]
            elif arrival_time + to_node.service_time > to_node.time_window[1]:
                start_service_time = (arrival_day + 1) * self.HOURS + to_node.time_window[0]
            else:
                start_service_time = arrival_cum_time
            # print(f'STARTING SERVICE at {start_service_time}, SERVICE TIME - {to_node.service_time}')
            end_time = start_service_time + to_node.service_time
            # print(f'FINISHED SERVICE {end_time}')
        return end_time - self.start_time

    def __repr__(self):
        return f'{self.vessel.name}: voyage TW - [{self.start_time}. {self.end_time}], ' \
               f'visited {self.route}, load - {self.load}'
