import random
import sys

from alns.Beans.installation import Installation
from alns.Beans.vessel import Vessel
from alns.Beans.voyage import Voyage
from alns.utils.utils import *
from alns.utils.distance_manager import DistanceManager
from alns.Beans.base import Base

import time


class Schedule:

    def __init__(self, vessels: list, installations: list, base: Base, schedule=None) -> None:
        self.vessels = vessels
        self.schedule = {}
        self.installations = installations
        self.base = base
        self.vessel_first_voyage_start_time = [-1 for n in range(len(self.vessels))]
        self.vessel_last_voyage_end_time = [0 for n in range(len(self.vessels))]
        if not schedule:
            self.distance_manager = DistanceManager(base, installations)
            self.generate_init_solution()
        else:
            self.schedule = schedule
        self.solution_cost = self.find_cost()

    def generate_init_solution(self):
        start = time.process_time()
        weekly_scenarios = build_weekly_departure_scnarios(self.installations)
        print("weekly scen -> " + str(time.process_time() - start))
        for i, daily_departure in enumerate(weekly_scenarios):
            voyage_pool = set()
            print("-------new day------")
            for inst_to_visit in daily_departure:
                start = time.process_time()

                voyage = self._get_free_voyage(voyage_pool, i, self.installations[inst_to_visit])

                print("get voyage -> " + str(time.process_time() - start))

                if voyage is None:
                    print("Cannot find feasible solution")
                    sys.exit()
                start = time.process_time()

                voyage.add_visit(self.installations[inst_to_visit], self.installations[inst_to_visit].deck_demand)

                print("add visit -> " + str(time.process_time() - start))
                self.vessel_last_voyage_end_time[voyage.vessel.name] = voyage.end_time

    def _get_free_voyage(self, voyage_pool: set, day: int, installation: Installation):
        local_voyage = None
        flag = False
        for possible_voyage in voyage_pool:
            vessel_first_start_time = self.vessel_first_voyage_start_time[possible_voyage.vessel.name]
            if possible_voyage.check_front_overlap(day, installation, vessel_first_start_time) and \
                    possible_voyage.deck_load + installation.deck_demand <= possible_voyage.vessel.deck_capacity:
                flag = True
                local_voyage = possible_voyage
                break
        if not voyage_pool or not flag:
            free_vessel = self._get_free_vessel(day, installation.deck_demand)
            if free_vessel is None:
                return None
            local_voyage = Voyage(vessel=free_vessel, distance_manager=self.distance_manager, base=self.base,
                                  start_day=day)
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

    def _get_free_vessel(self, day, demand):
        # check back_overlap and capacity
        possible_vessels = [self.vessels[i] for i in range(len(self.vessels)) if self.vessel_last_voyage_end_time[i]
                            < day * 24 + 8 and self.vessels[i].deck_capacity >= demand]
        if possible_vessels:
            return random.choice(possible_vessels)
        else:
            return None

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
