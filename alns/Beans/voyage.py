import numpy as np

from alns.Beans.vessel import Vessel
from alns.Beans.node import Installation, Base, Node
from typing import Type
from itertools import permutations

DAYS = 7
HOURS = 24
PERIOD_LENGTH = DAYS * HOURS
DEPARTURE_TIME = 16


class Voyage:

    def __init__(self,
                 base: Base, distance_manager, start_day: int):
        self.vessel = None
        self.route = []
        self.edges = []
        self.deck_load = 0
        self.start_time = start_day * HOURS + DEPARTURE_TIME
        self.start_day = start_day
        self.end_time = self.start_time
        self.base = base
        self.distance_manager = distance_manager
        # self.variable_cost = 0

    def __hash__(self):
        return hash((self.start_time, self.vessel.idx))

    def __repr__(self):
        route = '-'.join([str(i.idx) for i in self.route])
        route = f'{self.base.name}-{route}-{self.base.name}'
        return f'{route:>25}: [{self.start_time} - {self.end_time:.2f}]'

    def __str__(self):
        return f'{self.vessel.idx}:{self.start_day}'

    def __deepcopy__(self):
        copy = type(self)(self.base, self.distance_manager, self.start_day)
        copy.vessel = self.vessel
        copy.route = [r for r in self.route]
        return copy

    def check_overlap(self, other):
        """
        Checks if assigned voyage overlaps another voyage.
        :param other: another voyage
        :type other: Voyage
        :return: True if voyages overlap, otherwise False
        :rtype: bool
        """
        if self.start_day == other.start_day:
            return True
        # TODO: check if this is correct
        if other.end_time is None:
            e2 = other.earliest_end_time(speed=self.vessel.speed)
        else:
            e2 = other.end_time
        e2 = e2 + other.base.service_time
        s2 = other.start_time
        e1 = self.end_time + self.base.service_time
        s1 = self.start_time
        overlap1 = self.overlap(s1, e1, s2, e2)
        overlap2 = self.overlap(s1-PERIOD_LENGTH, e1-PERIOD_LENGTH, s2, e2)
        overlap3 = self.overlap(s1, e1, s2-PERIOD_LENGTH, e2-PERIOD_LENGTH)
        return overlap1 | overlap2 | overlap3

    @staticmethod
    def overlap(s1, e1, s2, e2):
        return (s2 < e1) & (s1 < e2)

    def earliest_end_time(self, speed):
        perm_routes = permutations(self.route)
        min_end_time = np.inf
        for route in perm_routes:
            end_time = self.calc_voyage_end_time(list(route), speed=speed)
            if end_time < min_end_time:
                min_end_time = end_time
        return min_end_time

    def improve_full_enum(self):
        perm_routes = permutations(self.route)
        min_end_time = np.inf
        best_route = None
        for route in perm_routes:
            end_time = self.calc_voyage_end_time(list(route))
            if end_time < min_end_time:
                min_end_time = end_time
                best_route = list(route)
        self.route = best_route
        self.edges = self.make_edges(best_route)
        # maybe make update method to avoid 'forgetting' to update essential parameters
        self.end_time = min_end_time

    def calc_voyage_end_time(self, route, speed=None):
        """
        Calculates time duration of the voyage.

        :return: time duration of the voyage in hours
        :rtype: float|int
        """
        if speed is None:
            if self.vessel is None:
                raise AttributeError('Voyage is not assigned to a vessel')
            speed = self.vessel.speed
        end_time = self.start_time
        edges = self.make_edges(route)
        for edge in edges[:-1]:
            to_node = edge[1]
            dist = edge[2]
            arrival_cum_time = end_time + (dist / speed)
            arrival_day = arrival_cum_time // HOURS
            arrival_time = arrival_cum_time % HOURS
            if arrival_time < to_node.time_window[0]:
                start_service_time = arrival_day * HOURS + to_node.time_window[0]
            elif arrival_time + to_node.service_time > to_node.time_window[1]:
                start_service_time = (arrival_day + 1) * HOURS + to_node.time_window[0]
            else:
                start_service_time = arrival_cum_time

            end_time = start_service_time + to_node.service_time
        end_time = end_time + edges[-1][2] / speed
        return end_time

    def total_wait_time(self):
        voyage_duration = self.end_time - self.start_time
        total_service_time = self.total_service_time()
        total_sailing_time = self.total_sailing_time()
        return voyage_duration - total_sailing_time - total_service_time

    def total_sailing_time(self):
        return sum([e[2] for e in self.edges])

    def total_service_time(self):
        return sum([i.service_time for i in self.route])

    def calc_variable_cost(self):
        voyage_duration = self.end_time - self.start_time
        total_service_time = self.total_service_time()
        total_sailing_time = self.total_sailing_time()
        total_waiting_time = voyage_duration - total_sailing_time - total_service_time
        variable_cost = (total_service_time+total_waiting_time) * self.vessel.fcw + total_sailing_time * self.vessel.fcs
        return np.around(variable_cost, 2)

    def add_visit(self, new_inst):
        """
        Adds new installation to visit in the voyage and recounts cost and end date.
        :param new_inst: new installation to be added
        :type new_inst: Installation
        :return:
        """
        self.route.append(new_inst)
        self.deck_load += new_inst.deck_demand

    # rename to remove inst from route
    # add remove visit method for Visit
    def remove_visit(self, installation: Installation):
        """
        removes installation from rout and recalculates time and cost of voyage
        :param installation:
        :return:
        """
        self.route.remove(installation)
        self.deck_load -= installation.deck_demand

    def update_edges(self):
        """
        Updates edges of the route
        :return:
        """
        self.edges = self.make_edges(self.route)

    def make_edges(self, route):
        """
        Edges of the route
        :return: list of edges (from_node, to_node, distance)
        :rtype: list[(Installation, Installation, float)]
        """
        route_with_base = [self.base] + route + [self.base]
        edges = zip(route_with_base[:-1], route_with_base[1:])
        return [(from_node, to_node, self.distance_manager.distance(from_node.name, to_node.name))
                for (from_node, to_node) in edges]

    def is_empty(self):
        return len(self.route) < 1

    def check_load_feasibility(self, vessel=None):
        if vessel is None:
            vessel = self.vessel
        return self.deck_load <= vessel.deck_capacity
