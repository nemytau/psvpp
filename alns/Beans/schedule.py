import random
import sys

import numpy as np

from alns.Beans.node import Installation, Base
from alns.Beans.vessel import Vessel
from alns.Beans.voyage import Voyage
from alns.utils.utils import daily_visits_from_departure_scenarios
from alns.utils.distance_manager import DistanceManager

import time
import pandas as pd

from tqdm import tqdm

DAYS = 7
HOURS = 24
PERIOD_LENGTH = DAYS * HOURS
DEPARTURE_TIME = 16


class Schedule:
    """
    Represents schedule. Provides functionality for building initial solution, validating solutions and calculating
    costs.
    """
    MAX_INST_PER_VOYAGE = 5
    MAX_ATTEMPTS_TO_INIT = 10

    def __init__(self, vessels: list[Vessel], installations: list[Installation], base: Base, schedule=None):
        self.vessels = vessels
        self.schedule = {}
        self.voyage_assignment = {}
        self.installations = installations
        self.base = base
        self.distance_manager = DistanceManager(base, installations)
        self.feasible = self.check_feasibility()
        if not schedule:
            self.generate_init_schedule()
        else:
            self.schedule = schedule

    def generate_init_schedule(self):
        extra_vessels = 0
        attempt_count = 0
        while not self.feasible:
            self.voyage_assignment = {v: [] for v in self.vessels}
            self.schedule = {day: [] for day in range(DAYS)}
            for _ in range(self.MAX_ATTEMPTS_TO_INIT):
                daily_visits = daily_visits_from_departure_scenarios(self.installations, DAYS)
                for day, day_visits in enumerate(daily_visits):
                    n_voyages = int(np.ceil(len(day_visits) / self.MAX_INST_PER_VOYAGE) + extra_vessels)
                    voyages = [Voyage(self.base,
                                      self.distance_manager,
                                      day) for _ in range(n_voyages)]
                    for inst in day_visits:
                        voyage = random.choice(voyages)
                        voyage.add_visit(inst)
                    self.assign_vessels(voyages, day)
                self.feasible = self.check_feasibility()
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
                can_insert = True
                if self.voyage_assignment[vessel]:
                    for assigned_voyage in self.voyage_assignment[vessel]:
                        if assigned_voyage.check_overlap(voyage):
                            can_insert = False
                            break
                if can_insert:
                    self.insert_voyage(voyage, vessel)
                    free_vessels.remove(vessel)
                    vessel.add_departure_day(day)
                    break

    def insert_voyage(self, voyage, vessel):
        """

        :param voyage:
        :type voyage:  Voyage
        :param vessel:
        :type vessel: Vessel
        :return:
        """
        voyage.vessel = vessel
        voyage.improve_full_enum()
        self.voyage_assignment[vessel].append(voyage)

    def _get_free_vessels(self, day):
        """

        :param day: day of interest
        :return: list of vessels without assigned voyages on the selected day
        :rtype: list[Vessel]
        """
        free_vessels = []
        for vessel in self.vessels:
            if day not in vessel.departure_days:
                free_vessels.append(vessel)
        return free_vessels

    def check_feasibility(self):
        if 'voyage_assignment' not in self.__dict__.keys():
            return False
        required_visits = {inst: inst.visit_frequency for inst in self.installations}
        actual_visits = {inst: 0 for inst in self.installations}
        for vessel, voyages in self.voyage_assignment.items():
            for voyage in voyages:
                for inst in voyage.route:
                    actual_visits[inst] += 1
        return required_visits == actual_visits

    def __repr__(self):
        schedule_to_str_list = []
        for vessel, voyages in self.schedule.items():
            vessel_voyages_str = '\n'.join([str(voyage) for voyage in voyages])
            schedule_to_str_list.append(f'Vessel {vessel}:\n{vessel_voyages_str}')
        return '\n'.join(schedule_to_str_list)

    def to_df(self):
        rows = []
        for vessel, voyages in self.voyage_assignment.items():
            for voyage in voyages:
                if voyage.end_time > PERIOD_LENGTH:
                    first_part = {
                        'Vessel': str(vessel),
                        'Route': '-'.join([i.name for i in voyage.route]),
                        'Start': voyage.start_time,
                        'End': PERIOD_LENGTH
                    }
                    second_part = {
                        'Vessel': str(vessel),
                        'Route': '-'.join([i.name for i in voyage.route]),
                        'Start': 0,
                        'End': voyage.end_time - PERIOD_LENGTH
                    }
                    rows.append(first_part)
                    rows.append(second_part)

                else:
                    row = {'Vessel': str(vessel),
                           'Route': '-'.join([i.name for i in voyage.route]),
                           'Start': voyage.start_time,
                           'End': voyage.end_time
                           }
                    rows.append(row)
        return pd.DataFrame(rows)
