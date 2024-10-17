from dataclasses import dataclass
from alns.Beans.voyage import Voyage
from alns.Beans.node import Installation


@dataclass
class VisitFreak:
    day: int
    vessel_idx: int
    inst_idx: int


@dataclass
class Visit:
    inst: Installation
    voyage: Voyage
