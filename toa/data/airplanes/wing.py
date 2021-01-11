from dataclasses import dataclass


@dataclass(frozen=False)
class Wings:
    area: float
    span: float
    mac: float
    sweep: float
    t_c: float

    def __post_init__(self):
        self.aspect_ratio = self.span ** 2 / self.area