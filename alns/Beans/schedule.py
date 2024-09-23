import random
import sys
from typing import List
import numpy as np
import logging
from alns.Beans.node import Installation, Base
from alns.Beans.vessel import Vessel
from alns.Beans.voyage import Voyage
from alns.Beans.visit import Visit
from alns.utils.utils import daily_visits_from_departure_scenarios
from alns.utils.distance_manager import DistanceManager
import time
import pandas as pd
from copy import deepcopy, copy
from tqdm import tqdm

DAYS = 7
HOURS = 24
PERIOD_LENGTH = DAYS * HOURS
DEPARTURE_TIME = 16

logger = logging.getLogger(__name__)

class Schedule:
    """
    Represents schedule. Provides functionality for building initial solution, validating solutions and calculating
    costs.
    """
    MAX_INST_PER_VOYAGE = 5
    MAX_ATTEMPTS_TO_INIT = 10

    def __init__(self, installations: List[Installation], vessels: List[Vessel], base: Base, schedule=None):
        self.vessels = vessels
        self.schedule = {}
        self.installations = installations
        self.base = base
        self.distance_manager = DistanceManager(base, installations)
        self.feasible = False
        if not schedule:
            self.generate_init_schedule()
        else:
            self.schedule = schedule
            self.feasible = self._check_feasibility()
            if not self.feasible:
                raise AttributeError('Passed schedule is not feasible')
        self.total_cost = self.calc_total_cost()

    def shallow_copy(self, ):
        """
        Creates shallow copy of schedule.
        :return:
        :rtype: Schedule
        """
        new_object = copy(self)
        new_object.schedule = {inst: [voyage.__deepcopy__() for voyage in voyages]
                               for inst, voyages in self.schedule.items()}
        return new_object

    def generate_init_schedule(self):
        extra_vessels = 0
        attempt_count = 0
        while not self.feasible:
            self.schedule = {v: [] for v in self.vessels}
            for _ in range(self.MAX_ATTEMPTS_TO_INIT):
                daily_visits = daily_visits_from_departure_scenarios(self.installations, DAYS)
                for day, day_visits in enumerate(daily_visits):
                    n_voyages = int(np.ceil(len(day_visits) / self.MAX_INST_PER_VOYAGE) + extra_vessels)
                    voyages = [Voyage(self.base,
                                      self.distance_manager,
                                      day) for _ in range(n_voyages)]
                    for inst in day_visits:
                        voyage = random.choice(voyages)
                        voyage.add_inst(inst)
                    self.assign_vessels(voyages, day)
                self.feasible = self._check_feasibility()
                attempt_count += 1
                if self.feasible:
                    break
            if not self.feasible:
                extra_vessels += 1

    def assign_vessels(self, voyages, day):
        free_vessels = self._get_free_vessels(day)
        free_vessels.sort()
        for voyage in voyages:
            if voyage.is_empty():
                continue
            for vessel in free_vessels:
                if not voyage.check_load_feasibility(vessel):
                    continue
                voyage.assign_vessel(vessel)
                can_insert = True
                if self.schedule[vessel]:
                    for assigned_voyage in self.schedule[vessel]:
                        if assigned_voyage.check_overlap_soft(voyage):
                            can_insert = False
                            break
                if can_insert:
                    self.append_voyage(voyage)
                    free_vessels.remove(vessel)
                    break

    # TODO: remove, not used
    def add_voyage(self, voyage, vessel):
        """

        :param voyage:
        :type voyage:  Voyage
        :param vessel:
        :type vessel: Vessel
        :return:
        """
        voyage.vessel = vessel
        voyage.improve_full_enum()
        self.schedule[vessel].append(voyage)

    def _get_free_vessels(self, day):
        """

        :param day: day of interest
        :return: list of vessels without assigned voyages on the selected day
        :rtype: list[Vessel]
        """
        # TODO: return only vessels that not on the voyage
        free_vessels = []
        for vessel in self.vessels:
            free_vessels.append(vessel)
        return free_vessels

    def check_feasibility_old(self):
        if 'schedule' not in self.__dict__.keys():
            return False
        required_visits = {inst: inst.visit_frequency for inst in self.installations}
        actual_visits = {inst: 0 for inst in self.installations}
        for vessel, voyages in self.schedule.items():
            for voyage in voyages:
                for inst in voyage.route:
                    actual_visits[inst] += 1
        return required_visits == actual_visits

    def __repr__(self):
        schedule_to_str_list = []
        for vessel, voyages in self.schedule.items():
            if voyages:
                vessel_voyages_str = '\n'.join([str(voyage) for voyage in voyages])
                schedule_to_str_list.append(f'{vessel}:\n{vessel_voyages_str}')
        return '\n'.join(schedule_to_str_list)

    def to_df_for_visualization(self):
        rows = []
        for vessel, voyages in self.schedule.items():
            for voyage in voyages:
                if voyage.end_time > PERIOD_LENGTH:
                    first_part = {
                        'Vessel': str(vessel),
                        'Route': '-'.join([str(i.idx) for i in voyage.route]),
                        'Start': voyage.start_time,
                        'End': PERIOD_LENGTH,
                        'Load': f'{voyage.deck_load}/{vessel.deck_capacity}'
                    }
                    second_part = {
                        'Vessel': str(vessel),
                        'Route': '-'.join([str(i.idx) for i in voyage.route]),
                        'Start': 0,
                        'End': voyage.end_time - PERIOD_LENGTH,
                        'Load': f'{voyage.deck_load}/{vessel.deck_capacity}'
                    }
                    rows.append(first_part)
                    rows.append(second_part)

                else:
                    row = {'Vessel': str(vessel),
                           'Route': '-'.join([str(i.idx) for i in voyage.route]),
                           'Start': voyage.start_time,
                           'End': voyage.end_time,
                           'Load': f'{voyage.deck_load}/{vessel.deck_capacity}'
                           }
                    rows.append(row)
        return pd.DataFrame(rows)

    def calc_total_cost(self):
        total_variable_cost = sum(
            [voyage.calc_variable_cost() for vessel_voyages in self.schedule.values() for voyage in vessel_voyages])
        total_fixed_cost = sum([v.cost for v in self.vessels if self.schedule[v]])
        return total_fixed_cost + total_variable_cost

    def visited_inst_voyage_list(self, voyages=None):
        """
        Returns list of tuples (inst, voyage) for all installations visited in the schedule
        :param voyages:
        :return:
        """
        if voyages is None:
            target_voyages = [voyage for vessel_voyages in self.schedule.values() for voyage in vessel_voyages]
        else:
            target_voyages = [voyage for vessel_voyages in self.schedule.values()
                       for voyage in vessel_voyages if voyage in voyages]
        inst_voyage_list = [(inst, voyage) for voyage in target_voyages for inst in voyage.route]
        return inst_voyage_list

    def remove_visit(self, visit):
        """

        :param visit:
        :type visit: Visit
        :return:
        """
        voyage = visit.voyage
        voyage.remove_inst(visit.inst)
        if voyage.is_empty():
            self.remove_voyage(voyage)
        else:
            voyage.improve_full_enum()

    def find_voyage(self, voyage):
        """

        :param voyage:
        :type voyage: Voyage
        :return:
        :rtype: Voyage
        """
        try :
            voyage = self.schedule[voyage.vessel][self.schedule[voyage.vessel].index(voyage)]
        except Exception as e:
            return None
        return voyage

    def find_voyage_on_day(self, vessel, day):
        """

        :param vessel:
        :type vessel: Vessel
        :param day:
        :type day: int
        :return:
        :rtype: Voyage
        """
        for voyage in self.schedule[vessel]:
            if voyage.start_day == day:
                return voyage
        return None

    def remove_visits_from_voyage(self, insts, voyage):
        if not isinstance(insts, list):
            insts = [insts]
        if voyage is None:
            logger.error('Passed empty voyage, location: schedule.remove_visits_from_voyage')
            raise Exception('Voyage is not found')
        the_voyage = self.find_voyage(voyage)
        if the_voyage is None:
            logger.warning('Voyage not found as object. Trying to find voyage by day')
            the_voyage = self.find_voyage_on_day(voyage.vessel, voyage.start_day)

        if the_voyage is None:
            logger.error('Voyage not found, location: schedule.remove_visits_from_voyage')
            raise Exception('Voyage not found')
        if len(insts) == len(the_voyage.route):
            self.remove_voyage(the_voyage)
        else:
            for inst in insts:
                the_voyage.remove_inst(inst)
            the_voyage.improve_full_enum()

    def remove_voyage(self, voyage):
        try:
            self.schedule[voyage.vessel].remove(voyage)
        except Exception as e:
            logger.error(f'Some error occurred while removing voyage from schedule: {e}, voyage-{voyage}')
        if voyage.is_empty():
            del voyage

    def find_voyages_containing_visit(self, inst):
        """
        :param inst:
        :type inst: Installation
        :return:
        :rtype: list[Voyage]
        """
        voyages = []
        for vessel, vessel_voyages in self.schedule.items():
            for voyage in vessel_voyages:
                if inst in voyage.route:
                    voyages.append(voyage)
        return voyages

    def check_for_replacement_overlap(self, new_voyage, old_voyage):
        for voyage in self.schedule[old_voyage.vessel]:
            if old_voyage != voyage and voyage.check_overlap_soft(new_voyage):
                return True
        return False

    def check_for_insertion_overlap(self, voyage, target_vessel):
        for target_voyage in self.schedule[target_vessel]:
            if voyage.check_overlap_soft(target_voyage):
                return True
        return False

    def add_idle_vessel(self, vessel=None):
        # TODO: Choose largest vessel
        if vessel is None:
            for key, value in self.schedule.items():
                if not value:
                    vessel = key
                    break
        self.add_empty_voyages(vessel)

    def is_vessel_idle(self, vessel):
        if not self.schedule[vessel]:
            return True
        else:
            for voyage in self.schedule[vessel]:
                if not voyage.is_empty():
                    return False
        return True

    def add_empty_voyages(self, vessel):
        for day in range(DAYS):
            # NOTE: Creating new instance - costly operation
            voyage = Voyage(self.base, self.distance_manager, day)
            voyage.end_time = voyage.start_time
            self.insert_voyage(voyage, vessel)

    def insert_voyage(self, voyage, vessel):
        can_insert = True
        if self.schedule[vessel]:
            for v in self.schedule[vessel]:
                # NOTE: Replaced soft overlap with hard overlap
                if v.check_overlap_hard(voyage):
                    can_insert = False
                    break
        if can_insert:
            voyage.vessel = vessel
            self.schedule[vessel].append(voyage)
            return True
        return False

    def append_voyage(self, voyage):
        self.schedule[voyage.vessel].append(voyage)

    def insert_idle_vessel_and_add_empty_voyages(self):
        for vessel in self.vessels:
            if self.schedule[vessel]:
                self.add_empty_voyages(vessel)
        self.add_idle_vessel()

    def _check_feasibility(self):
        if 'schedule' not in self.__dict__.keys():
            return False
        demand_visits = {inst: inst.visit_frequency for inst in self.installations}
        capacity_violated = False
        spread_violated = False
        double_visited = False
        for vessel, voyages in self.schedule.items():
            for voyage in voyages:
                if not voyage.check_load_feasibility():
                    capacity_violated = True
                for inst in voyage.route:
                    demand_visits[inst] -= 1
        demand_is_covered = min([v == 0 for v in demand_visits.values()])
        return not capacity_violated and demand_is_covered

    def _check_demand_coverage(self):
        demand_visits = {inst: inst.visit_frequency for inst in self.installations}
        for vessel, voyages in self.schedule.items():
            for voyage in voyages:
                for inst in voyage.route:
                    demand_visits[inst] -= 1
        print(demand_visits)

    def update_feasibility(self):
        self.feasible = self._check_feasibility()

    def update_cost(self):
        self.total_cost = self.calc_total_cost()

    def update(self):
        self.update_feasibility()
        self.update_cost()

    def drop_empty_voyages(self):
        """
        Removes empty voyages from schedule
        :param sch:
        :type sch: Schedule
        :return:
        """
        empty_voyages = []
        for vessel, voyages in self.schedule.items():
            for voyage in voyages:
                if len(voyage.route) == 0:
                    empty_voyages.append((vessel, voyage))
        for vessel, voyage in empty_voyages:
            self.remove_voyage(voyage)

    def flattened_voyages(self):
        return [voyage for voyages in self.schedule.values() for voyage in voyages]

    def get_nonempty_vessels(self):
        return [vessel for vessel, voyages in self.schedule.items() if voyages]

    def calculate_reassignment_overlap(self, voyage, target_vessel):
        """
        Calculates the overlap between a voyage and a target vessel
        :param voyage:
        :type voyage: Voyage
        :param target_vessel:
        :type target_vessel: Vessel
        :return:
        :rtype: int
        """
        overlap = 0
        for v in self.schedule[target_vessel]:
            overlap += voyage.calculate_overlap(v)
        return overlap

    def voyages_to_shrink_to_fit(self, voyage_to_move, target_v):
        """
        Finds voyages that need to be shrunk to fit the new voyage.
        From overlapping voyages, all voyages except of the last one are chosen.
        :param voyage_to_move: Voyage from another vessel to be inserted
        :type voyage_to_move: Voyage
        :param target_v: Target vessel
        :type target_v: Vessel
        :return:
        """
        voyages_to_shrink = []
        # NOTE: voyage_to_move could be modified in the schedule, but voyage_to_move is not updated
        actual_voyage_to_move = self.find_voyage_on_day(voyage_to_move.vessel, voyage_to_move.start_day)
        for voyage in self.schedule[target_v]:
            if voyage.check_overlap_hard(voyage_to_move):
                voyages_to_shrink.append(voyage)
        voyages_to_shrink.append(actual_voyage_to_move)
        voyages_to_shrink.sort(key=lambda x: x.start_time)
        return voyages_to_shrink[:-1]

    def reassign_voyage_to_vessel(self, voyage_to_move, target_v):
        """
        Reassigns voyage to another vessel
        :param voyage_to_move: Voyage to be moved
        :type voyage_to_move: Voyage
        :param target_v: Target vessel
        :type target_v: Vessel
        :return:
        """
        self.remove_voyage(voyage_to_move)
        is_inserted = self.insert_voyage(voyage_to_move, target_v)
        if not is_inserted:
            raise Exception('Voyage could not be inserted')

    def force_reassign_voyages(self, origin_v, target_v):
        """
        Reassigns voyages to another vessel. Reassignment should be feasible.
        :param origin_v: Origin vessel
        :type origin_v: Vessel
        :param target_v: Target vessel
        :type target_v: Vessel
        :return:
        """
        voyages_to_move = self.schedule[origin_v]
        for voyage in voyages_to_move:
            voyage.vessel = target_v
        self.schedule[target_v].extend(voyages_to_move)
        self.schedule[origin_v] = []

    def relocate_visit(self, inst, from_voyage, to_voyage):
        """
        Relocates visit from one voyage to another
        :param inst: Installation to be relocated
        :type inst: Installation
        :param from_voyage: Voyage to relocate from
        :type from_voyage: Voyage
        :param to_voyage: Voyage to relocate to
        :type to_voyage: Voyage
        :return:
        """
        # Maybe voyages should be taken from the schedule
        origin_voyage = self.find_voyage_on_day(from_voyage.vessel, from_voyage.start_day)
        target_voyage = self.find_voyage_on_day(to_voyage.vessel, to_voyage.start_day)
        origin_voyage.remove_visit(inst)
        target_voyage.insert_visit(inst)
        if origin_voyage.is_empty():
            self.remove_voyage(origin_voyage)

    def remove_inst_from_voyage_route(self, inst, voyage):
        """
        Removes installation from route in the voyage
        :param inst: Installation to be removed
        :type inst: Installation
        :param voyage: Voyage
        :type voyage: Voyage
        :return:
        """
        actual_voyage = self.find_voyage(voyage)
        voyage_len = len(actual_voyage.route)
        actual_voyage.remove_inst(inst)
        if voyage_len - len(actual_voyage.route) != 1:
            raise Exception('Visit was not removed')

    def swap_visits(self, visit1, visit2):
        """
        Swaps two visits in the schedule
        :param visit1: First visit
        :type visit1: tuple[Installation, Voyage]
        :param visit2: Second visit
        :type visit2: tuple[Installation, Voyage]
        :return:
        """
        voyage1 = self.find_voyage(visit1[1])
        voyage2 = self.find_voyage(visit2[1])
        if voyage1 is None or voyage2 is None:
            # TODO: Add to log
            return
        # TODO: Remove check, if method works well
        voyage1_len = len(voyage1.route)
        # TODO: Remove check, if method works well
        voyage2_len = len(voyage2.route)
        if voyage1.is_on_the_route(visit2[0]) or voyage2.is_on_the_route(visit1[0]):
            return
        voyage1.remove_inst(visit1[0])
        voyage2.remove_inst(visit2[0])
        # TODO: Remove check, if method works well
        if voyage1_len - len(voyage1.route) != 1 or voyage2_len - len(voyage2.route) != 1:
            raise Exception('Visit was not removed')
        voyage1.insert_visit(visit2[0])
        voyage2.insert_visit(visit1[0])
        # TODO: Remove check, if method works well
        if voyage1_len - len(voyage1.route) != 0 or voyage2_len - len(voyage2.route) != 0:
            raise Exception('Visit was not added')

    def swap_visits_tuple_repr(self, visit1, visit2):
        """
        Swaps two visits in the schedule. Visits are represented as tuples (inst, vessel, start_day)
        :param visit1: First visit
        :type visit1: tuple[Installation, Vessel, int]
        :param visit2: Second visit
        :type visit2: tuple[Installation, Vessel, int]
        :return:
        """
        voyage1 = self.find_voyage_on_day(visit1[1], visit1[2])
        voyage2 = self.find_voyage_on_day(visit2[1], visit2[2])
        if voyage1 is None or voyage2 is None:
            # TODO: Add to log
            return
        if voyage1.is_on_the_route(visit2[0]) or voyage2.is_on_the_route(visit1[0]):
            return
        # TODO: Remove check, if method works well
        voyage1_len = len(voyage1.route)
        voyage2_len = len(voyage2.route)

        voyage1.remove_inst(visit1[0])
        voyage2.remove_inst(visit2[0])

        # TODO: Remove check, if method works well
        if voyage1_len - len(voyage1.route) != 1 or voyage2_len - len(voyage2.route) != 1:
            raise Exception('Visit was not removed')

        voyage1.insert_visit(visit2[0])
        voyage2.insert_visit(visit1[0])
        # TODO: Remove check, if method works well
        if voyage1_len - len(voyage1.route) != 0 or voyage2_len - len(voyage2.route) != 0:
            raise Exception('Visit was not added')

    def visits_list_plain(self):
        """
        Returns list of visits in the schedule as tuples (inst, vessel, start_day)
        :return:
        """
        visits = []
        for vessel, voyages in self.schedule.items():
            for voyage in voyages:
                for inst in voyage.route:
                    visits.append((inst, vessel, voyage.start_day))
        return visits
