from dataclasses import dataclass


@dataclass(frozen=False)
class AirplaneData:
    S: float
    cbar: float
    Iy: float
    CDmin: float
    kCDi: float
    CL0: float
    CLa: float
    CLde: float
    CLq: float
    Cm0: float
    Cma: float
    Cmde: float
    Cmq: float
    xn: float
    xm: float
    zmn: float
    zt: float
