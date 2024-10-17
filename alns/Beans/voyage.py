import numpy as np
import pandas as pd

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
        return f'FL:{self.vessel.deck_capacity - self.deck_load} {route:>20}: [{self.start_time} - {self.end_time:.2f}]'

    def __str__(self):
        return f'{self.vessel.idx}:{self.start_day}'

    def __deepcopy__(self):
        copy = type(self)(self.base, self.distance_manager, self.start_day)
        copy.vessel = self.vessel
        copy.route = [r for r in self.route]
        copy.end_time = self.end_time
        copy.deck_load = self.deck_load
        return copy

    def __eq__(self, other):
        if isinstance(other, Voyage):
            return self.vessel.idx == other.vessel.idx \
                   and self.start_day == other.start_day \
                   and self.route == other.route
        return False

    # TODO: merge with check_overlap_hard method by adding a parameter to ignore empty voyages.
    def check_overlap_soft(self, other):
        """
        Checks if assigned voyage overlaps another voyage. Ignores empty voyages.
        :param other: another voyage
        :type other: Voyage
        :return: True if voyages overlap, otherwise False
        :rtype: bool
        """
        if self.start_day == other.start_day:
            return True
        if len(self.route) == 0 or len(other.route) == 0:
            return False
        return self.check_overlap(self, other)

    def check_overlap_hard(self, other):
        """
        Checks if assigned voyage overlaps another voyage.
        :param other: another voyage
        :type other: Voyage
        :return: True if voyages overlap, otherwise False
        :rtype: bool
        """
        if self.start_day == other.start_day:
            return True
        return self.check_overlap(self, other)

    @staticmethod
    def check_overlap(voyage1, voyage2):
        if voyage2.end_time is None:
            e2 = voyage2.earliest_end_time(speed=voyage1.vessel.speed)
        else:
            e2 = voyage2.end_time
        e2 = e2 + voyage2.base.service_time
        s2 = voyage2.start_time
        e1 = voyage1.end_time + voyage1.base.service_time
        s1 = voyage1.start_time
        overlap1 = voyage1.overlap(s1, e1, s2, e2)
        overlap2 = voyage1.overlap(s1 - PERIOD_LENGTH, e1 - PERIOD_LENGTH, s2, e2)
        overlap3 = voyage1.overlap(s1, e1, s2 - PERIOD_LENGTH, e2 - PERIOD_LENGTH)
        return overlap1 | overlap2 | overlap3

    @staticmethod
    def overlap(s1, e1, s2, e2):
        # NOTE: added equal sign to check if voyages start at the same time, may be rounding problem
        return (s2 <= e1) & (s1 <= e2)

    def is_on_the_route(self, inst: Installation):
        return inst in self.route

    def earliest_end_time(self, speed):
        perm_routes = permutations(self.route)
        min_end_time = np.inf
        for route in perm_routes:
            end_time = self.calc_voyage_end_time(list(route), speed=speed)
            if end_time < min_end_time:
                min_end_time = end_time
        return min_end_time

    def assign_vessel(self, vessel: Vessel):
        # TODO: kostyyl, unassigned voyages have vessels in attributes to reduce number of calculations
        if self.vessel is None:
            self.vessel = vessel
            self.improve_full_enum()
        else:
            self.vessel = vessel

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
            if arrival_time < to_node.adjTW[0]:
                start_service_time = arrival_day * HOURS + to_node.adjTW[0]
            elif arrival_time > to_node.adjTW[1]:
                start_service_time = (arrival_day + 1) * HOURS + to_node.adjTW[0]
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
        edges = self.make_edges(self.route)
        return sum([e[2]/self.vessel.speed for e in edges])

    def total_service_time(self):
        return sum([i.service_time for i in self.route])

    def calc_variable_cost(self):
        voyage_duration = self.end_time - self.start_time
        total_service_time = self.total_service_time()
        total_sailing_time = self.total_sailing_time()
        total_waiting_time = voyage_duration - total_sailing_time - total_service_time
        variable_cost = (total_service_time + total_waiting_time) * self.vessel.fcw + total_sailing_time * self.vessel.fcs
        return np.around(variable_cost, 2)

    def add_inst(self, new_inst):
        """
        Adds new installation to visit in the voyage and recounts cost and end date.
        :param new_inst: new installation to be added
        :type new_inst: Installation
        :return:
        """
        self.route.append(new_inst)
        self.deck_load += new_inst.deck_demand

    # rename to remove inst from route
    def remove_inst(self, installation: Installation):
        """
        removes installation from rout and recalculates time and cost of voyage
        :param installation:
        :return:
        """
        self.route.remove(installation)
        self.deck_load -= installation.deck_demand

    def insert_visit(self, inst):
        if inst in self.route:
            raise ValueError('Installation is already in the route')
            return
        self.add_inst(inst)
        self.improve_full_enum()

    def remove_visit(self, inst):
        self.remove_inst(inst)
        self.improve_full_enum()

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

    def calculate_overlap(self, other):
        """
        Calculates overlap between two voyages
        :param other: other voyage
        :type other: Voyage
        :return: overlap between two voyages
        :rtype: float
        """
        overlap = 0
        if self.check_overlap(self, other):
            e1 = self.end_time + self.base.service_time
            s1 = self.start_time
            e2 = other.end_time + other.base.service_time
            s2 = other.start_time
            overlap = np.min([e1, e2]) - np.max([s1, s2])
            if overlap < 0:
                overlap = (e1 - s1) + (e2 - s2) - (PERIOD_LENGTH + overlap)
        return overlap

    def get_route_stages_df(self):
        """
        Returns a DataFrame with information on different stages of the route performed by the vessel.

        Each row represents a distinct route stage and includes:
        - Action start time
        - Action end time
        - Type of action (sailing, servicing installation, waiting).
        :return:
        """

        def split_stage_crossing_horizon(stage):
            """
            Splits a stage that crosses the horizon into two stages.
            :param stage: stage that crosses the horizon
            :return: two stages
            """
            first_stage = {'start_time': stage['start_time'],
                           'end_time': PERIOD_LENGTH,
                           'action': stage['action'],
                           'description': stage['description']}
            second_stage = {'start_time': 0,
                            'end_time': stage['end_time'] % PERIOD_LENGTH,
                            'action': stage['action'],
                            'description': stage['description']}
            return first_stage, second_stage

        def offset_stage_out_of_horizon(stage):
            """
            Offsets a stage that crosses the horizon out of the horizon.
            :param stage: stage that crosses the horizon
            :return: offset stage
            """
            return {'start_time': stage['start_time'] % PERIOD_LENGTH,
                    'end_time': stage['end_time'] % PERIOD_LENGTH,
                    'action': stage['action'],
                    'description': stage['description']}

        stages = []
        current_time = self.start_time - self.base.service_time
        edges = self.make_edges(self.route)
        for i, edge in enumerate(edges):
            service_i = {'start_time': current_time,
                         'end_time': current_time + edge[0].service_time,
                         'action': 'Service'if edge[0].idx != 0 else 'Service at base',
                         'description': f'{edge[0].idx}'}
            current_time = service_i['end_time']
            sailing_ij = {'start_time': current_time,
                          'end_time': current_time + self.distance_manager.distance(edge[0].name,
                                                                                    edge[1].name) / self.vessel.speed,
                          'action': 'Sailing',
                          'description': f'{edge[0].idx}-{edge[1].idx}'}
            current_time = sailing_ij['end_time']
            arrival_time = current_time % HOURS
            arrival_day = current_time // HOURS
            if arrival_time < edge[1].adjTW[0]:
                end_wait_time = arrival_day * HOURS + edge[1].adjTW[0]
            elif arrival_time > edge[1].adjTW[1]:
                end_wait_time = (arrival_day + 1) * HOURS + edge[1].adjTW[0]
            else:
                end_wait_time = np.nan
            waiting_j = {'start_time': current_time,
                         'end_time': end_wait_time,
                         'action': 'Waiting' if edge[1].idx != 0 else 'Waiting at base',
                         'description': f'{edge[1].idx}'} if not np.isnan(end_wait_time) else None
            current_time = end_wait_time if not np.isnan(end_wait_time) else current_time
            stages.extend([service_i, sailing_ij, waiting_j])
        stages = [stage for stage in stages if stage is not None]
        stages_splitted = []
        for stage in stages:
            if (stage['start_time'] < PERIOD_LENGTH) & (stage['end_time'] > PERIOD_LENGTH):
                stages_splitted.extend(split_stage_crossing_horizon(stage))
            else:
                stages_splitted.append(stage)
        stages = stages_splitted
        stages = [offset_stage_out_of_horizon(stage) if stage['start_time'] > PERIOD_LENGTH else stage for stage in
                  stages]
        stages_df = pd.DataFrame(stages)
        stages_df['Vessel'] = str(self.vessel.name)
        return stages_df
