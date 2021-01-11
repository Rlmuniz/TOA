from dataclasses import dataclass


@dataclass(frozen=False)
class Flaps:
    type: str
    area: float
    bf_b: float
    sf_s: float
    cf_c: float
    lambda_f: float