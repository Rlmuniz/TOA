from dataclasses import dataclass


@dataclass(frozen=False)
class Limits:
    MTOW: float