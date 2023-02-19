from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Base:
    name: str
    service_time: float
    time_window: tuple[int, int]
