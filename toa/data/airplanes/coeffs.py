from dataclasses import dataclass


@dataclass(frozen=False)
class AerodynamicCoefficients:
    CL0: float
    CLalpha: float
    CLde: float
    CLq: float
    Cm0: float
    Cmalpha: float
    Cmde: float
    Cmq: float