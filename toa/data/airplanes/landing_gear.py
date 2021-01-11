from dataclasses import dataclass


@dataclass(frozen=False)
class LandingGear:
    x_ng: float
    z_ng: float
    nw_ng: int
    x_mg: float
    z_mg: float
    nw_mg: int