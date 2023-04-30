import random
import sys

from alns.Beans2.installation import Installation
from alns.Beans2.vessel import Vessel
from alns.Beans2.voyage import Voyage
from alns.utils.utils import *
from alns.utils.distance_manager import DistanceManager
from alns.Beans2.base import Base


import time


class Schedule:

    def __init__(self, vessels: list, installations: list, base: Base, distance_manager: DistanceManager, schedule=None):
        self.vessels = vessels
        # self.weighted_vessels = list(zip(vessels,[1 for _ in range(len(vessels))]))
        self.schedule = {}
        self.installations = installations
        self.base = base
        self.vessel_first_voyage_start_time = [-1 for n in range(len(self.vessels))]
        self.vessel_last_voyage_end_time = [0 for n in range(len(self.vessels))]
        self.distance_manager = distance_manager
        if not schedule:
            self.generate_init_solution()
        else:
            self.schedule = schedule
        self.solution_cost = self.find_cost()

    def generate_init_solution(self):
        start = time.process_time()
        weekly_scenarios = daily_visits_from_departure_scenarios(self.installations)
        # weekly_scenarios = [[3, 4, 6], [0, 1, 7, 8], [2, 3, 4, 5], [6, 9], [4], [0, 7], [2]]
        print("weekly scen -> " + str(time.process_time() - start))
        for day, daily_departure in enumerate(weekly_scenarios):
            voyage_pool = set()
            print("-------new day------")
            for inst_to_visit in daily_departure:
                start = time.process_time()

                voyage = self._get_free_voyage(voyage_pool, day, inst_to_visit)

                print("get voyage -> " + str(time.process_time() - start))

                if voyage is None:
                    print("Cannot find feasible solution")
                    sys.exit()
                start = time.process_time()

                voyage.add_visit(inst_to_visit, inst_to_visit.deck_demand)

                print("add visit -> " + str(time.process_time() - start))
                self.vessel_last_voyage_end_time[voyage.vessel.name] = voyage.end_time

    def _get_free_voyage(self, voyage_pool: set, day: int, installation: Installation):
        local_voyage = None
        voyage_was_found = False
        for possible_voyage in voyage_pool:
            vessel_first_start_time = self.vessel_first_voyage_start_time[possible_voyage.vessel.name]
            if possible_voyage.check_front_overlap(installation, vessel_first_start_time) and \
                    possible_voyage.deck_load + installation.deck_demand <= possible_voyage.vessel.deck_capacity:
                voyage_was_found = True
                local_voyage = possible_voyage
                break
        if not voyage_pool or not voyage_was_found:
            free_vessels = self._get_free_vessels(day, installation.deck_demand)
            if free_vessels is None:
                return None
            free_vessel = None
            for free_vessel in free_vessels:
                local_voyage = Voyage(vessel=free_vessel, distance_manager=self.distance_manager, base=self.base,
                                      start_day=day)
                if local_voyage.check_front_overlap(installation, self.vessel_first_voyage_start_time[free_vessel.name]):
                    break
            self.add_voyage(free_vessel, local_voyage)
            if self.vessel_first_voyage_start_time[free_vessel.name] == -1:
                self.vessel_first_voyage_start_time[free_vessel.name] = local_voyage.start_time
            voyage_pool.add(local_voyage)
        return local_voyage

    def add_voyage(self, vessel: Vessel, voyage: Voyage):
        if vessel.name in self.schedule:
            self.schedule[vessel.name].append(voyage)
        else:
            self.schedule[vessel.name] = [voyage]

    def remove_voyage(self, vessel: Vessel, voyage: Voyage):
        self.schedule[vessel.name].remove(voyage)

    def _get_free_vessels(self, day, demand):
        # check back_overlap and capacity
        possible_vessels = [self.vessels[i] for i in range(len(self.vessels)) if self.vessel_last_voyage_end_time[i]
                            < day * 24 + 8 and self.vessels[i].deck_capacity >= demand]
        # if possible_vessels:
        #     return random.choice(possible_vessels)
        # else:
        #     return None
        return possible_vessels

    def remove_visit(self, voyage: Voyage, installation):
        voyage.remove_visit(installation)
        if not voyage.route:
            # self.schedule[voyage.vessel.name].remove(voyage)
            self.remove_voyage(voyage.vessel, voyage)
            if not self.schedule[voyage.vessel.name]:
                self.schedule.pop(voyage.vessel.name)

    def find_cost(self):
        overall_cost = 0
        for vessel_name in self.schedule:
            overall_cost += self.schedule[vessel_name][0].vessel.cost
            for voyage in self.schedule[vessel_name]:
                overall_cost += voyage.variable_cost
        return overall_cost
