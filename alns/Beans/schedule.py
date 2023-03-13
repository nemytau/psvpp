from alns.Beans.installation import Installation
from alns.Beans.vessel import Vessel
from alns.Beans.voyage import Voyage
from alns.utils.utils import *
from alns.utils.distance_manager import DistanceManager
from alns.Beans.base import Base


class Schedule:

    def __init__(self, vessels: list, installations: list, base:Base, schedule=None) -> None:
        if schedule is None:
            schedule = []
        self.vessels = vessels
        self.installations = installations
        self.base = base
        if not schedule:
            self.schedule = self.generate_init_solution()
            self.distance_manager = DistanceManager(base, installations)
        else:
            self.schedule = schedule

    def generate_init_solution(self):
        voyage_pool = set()
        schedule = {}
        weekly_scenarios = build_weekly_departure_scnarios(self.installations)
        for i,daily_departure in enumerate(weekly_scenarios):
            for inst_to_visit in daily_departure:
                voyage = self._get_free_voyage(voyage_pool, i)
                if voyage.deck_load + self.installations[inst_to_visit].deck_demand <= voyage.vessel.deck_capacity:
                     voyage.add_visit(self.installations[inst_to_visit],self.installations[inst_to_visit].deck_demand)
        for voyage in voyage_pool:
            if voyage.vessel.name in schedule:
                schedule[voyage.vessel.name].append(voyage)
            else:
                schedule[voyage.vessel.name] = [voyage]
        return schedule

    def _get_free_voyage(self, voyage_pool, day):
        voyage = None
        flag = False
        for possible_voyage in voyage_pool:
            if possible_voyage.check_overlap(day):
                flag = True
                voyage = possible_voyage
                break
        if len(voyage_pool) == 0 or not flag:
            voyage = Voyage(vessel=self.vessels[0],distance_manager=self.distance_manager, base=self.base, start_day=day) #TODO fix self.vessels[0]
            voyage_pool.add(voyage)

        return voyage
