from dataclasses import dataclass


@dataclass(frozen=False)
class Inertia:
    Iy: float