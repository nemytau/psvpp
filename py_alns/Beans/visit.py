from dataclasses import dataclass
from py_alns.Beans.voyage import Voyage
from py_alns.Beans.node import Installation


@dataclass
class VisitFreak:
    day: int
    vessel_idx: int
    inst_idx: int


@dataclass
class Visit:
    inst: Installation
    voyage: Voyage
