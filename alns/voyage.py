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
        self.end_time = self.calc_end_time(self.start_time, self.route)

    def calc_end_time(self, start_time, route):
        """
        Calculates end time of the voyage.

        :param start_time: starting time of the voyage
        :type start_time: float
        :param route: route of the voyage
        :type route: Route
        :return: end time of the voyage
        :rtype: float
        """
        end_time = start_time
        # print(f'START {start_time}')
        for edge in route.edges():
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
        return end_time % self.PERIOD_LENGTH

    def __repr__(self):
        return f'{self.vessel.name}: voyage TW - [{self.start_time}. {self.end_time}], ' \
               f'visited {self.route}'
