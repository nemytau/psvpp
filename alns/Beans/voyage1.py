from alns.Beans.vessel1 import Vessel
from alns.Beans.installation1 import Installation
from alns.Beans.base1 import Base

DAYS = 7
HOURS = 24
PERIOD_LENGTH = 168
DEPARTURE_PERIOD = 8


class Voyage:

    def __init__(self,
                 vessel: Vessel, base: Base, distance_manager, start_day: int):
        self.vessel = vessel
        self.route = []
        self.deck_load = 0
        self.start_time = start_day * HOURS + DEPARTURE_PERIOD
        self.voyage_length = 0
        self.end_time = None
        self.base = base
        self.distance_manager = distance_manager
        self.variable_cost = 0
        # self.load = self.calc_load()

    def calc_voyage_duration(self, route):
        """
        Calculates time length of the voyage.

        :return: time length of the voyage in hours
        :rtype: float
        """
        end_time = self.start_time
        length = 0
        for edge in self.edges([self.base] + route + [self.base]):
            to_node = edge[1]
            dist = edge[2]
            length += dist
            # arrival_cum_time = (end_time + dist / self.vessel.speed)
            arrival_cum_time = end_time + (dist / self.vessel.speed)

            # print(f'ARRIVED {arrival_cum_time}, SAILED {dist}, TW: {to_node.time_window}')
            arrival_day = arrival_cum_time // HOURS
            arrival_time = arrival_cum_time % HOURS
            if arrival_time < to_node.time_window[0]:
                start_service_time = arrival_day * HOURS + to_node.time_window[0]
            elif arrival_time + to_node.service_time > to_node.time_window[1]:
                start_service_time = (arrival_day + 1) * HOURS + to_node.time_window[0]
            else:
                start_service_time = arrival_cum_time
            # print(f'STARTING SERVICE at {start_service_time}, SERVICE TIME - {to_node.service_time}')
            end_time = start_service_time + to_node.service_time
            # print(f'FINISHED SERVICE {end_time}')
        return end_time

    def calc_variable_cost(self):
        return (self.end_time-self.start_time)*self.vessel.fuel_consumption

    def add_visit(self, new_inst, load):
        """
        Adds new installation to visit in the voyage and recounts cost and end date.
        :param new_inst: new installation to be added
        :return: None
        """
        self.route.append(new_inst)
        self.deck_load += load
        self.end_time = self.calc_voyage_duration(self.route)
        self.variable_cost = self.calc_variable_cost()

    def remove_visit(self, installation: Installation):
        """
        removes installation from rout and recalculates time and cost of voyage
        :param installation:
        :return:
        """
        self.route.remove(installation)
        self.deck_load -= installation.deck_demand
        self.end_time = self.calc_voyage_duration(self.route)
        self.variable_cost = self.calc_variable_cost()

    def edges(self, route):
        """
        Edges of the route
        :return: list of edges [from_node, to_node, distance]
        :rtype: list[list[Installation, Installation, float]]
        """
        edges = zip(route[:-1], route[1:])
        return [(from_node, to_node, self.distance_manager.distance(from_node.name, to_node.name))
                for (from_node, to_node) in edges]

    def check_front_overlap(self, installation, vessel_first_start_time):
        """
        Check if possible to choose this voyage for day 'day'

        :param vessel_first_start_time:
        :param installation: installation we would like to add
        :return: True - possible to use this voyage, False - otherwise.
        """
        route = self.route.copy()
        route.append(installation)
        new_end_time = self.calc_voyage_duration(route)
        return new_end_time - PERIOD_LENGTH < vessel_first_start_time

    # def update_voyage_by_new_route(self, route):
    #     length, new_end_time = self.calc_voyage_length_and_duration(route)
    #     self.voyage_length = length
    #     self.end_time = new_end_time
    #     self.variable_cost = self.calc_variable_cost()
