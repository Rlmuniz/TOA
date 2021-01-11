from dataclasses import dataclass

from toa.data.airplanes.coeffs import AerodynamicCoefficients
from toa.data.airplanes.engines import Engines
from toa.data.airplanes.flaps import Flaps
from toa.data.airplanes.inertia import Inertia
from toa.data.airplanes.landing_gear import LandingGear
from toa.data.airplanes.limits import Limits
from toa.data.airplanes.polar import Polar
from toa.data.airplanes.wing import Wings


@dataclass(frozen=False)
class Airplanes:
    wing: Wings
    flap: Flaps
    engine: Engines
    limits: Limits
    polar: Polar
    landing_gear: LandingGear
    inertia: Inertia
    coeffs: AerodynamicCoefficients
