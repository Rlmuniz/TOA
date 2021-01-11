from dataclasses import dataclass


@dataclass(frozen=False)
class Polar:
    CD0: float
    k: float
    e: float