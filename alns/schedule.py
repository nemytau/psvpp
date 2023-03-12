from .fleet import Fleet
from .installation import Installation


class VesselSchedule:
    def __init__(self,
                 vessel,
                 voyages):
        self.vessel = vessel
        self.voyages = voyages


class Schedule:
    def __init__(self,
                 insts,
                 fleet,
                 base,
                 v_schedules
                 ):
        self.insts = insts
        self.fleet = fleet
        self.base = base
        self.v_schedules = v_schedules
