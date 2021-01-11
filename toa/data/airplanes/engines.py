from dataclasses import dataclass
from dataclasses import field
from math import sqrt


@dataclass(frozen=False)
class Engines:
    max_thrust_sl: float
    bypass_ratio: float
    zpos: float
    mount: str
    num_motors: int
    cff1: float
    cff2: float
    cff3: float
    G0: float = field(init=False)
    k1: float = field(init=False)
    k2: float = field(init=False)

    def __post_init__(self):
        self.G0 = 0.06 * self.bypass_ratio + 0.64
        self.k1 = 0.377 * (1 + self.bypass_ratio) / sqrt((1 + 0.82 * self.bypass_ratio) * self.G0)
        self.k2 = 0.23 + 0.19 * sqrt(self.bypass_ratio)