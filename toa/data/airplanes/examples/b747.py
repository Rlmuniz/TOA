from toa.data.airplanes.airplanes import Airplanes
from toa.data.airplanes.coeffs import AerodynamicCoefficients
from toa.data.airplanes.engines import Engines
from toa.data.airplanes.flaps import Flaps
from toa.data.airplanes.inertia import Inertia
from toa.data.airplanes.landing_gear import LandingGear
from toa.data.airplanes.limits import Limits
from toa.data.airplanes.polar import Polar
from toa.data.airplanes.wing import Wings

b747 = Airplanes(
        wing=Wings(area=525.6,
                   span=64.4,
                   mac=9.68,
                   sweep=37.5,
                   t_c=0.094),
        flap=Flaps(type="triple-slotted",
                   area=78.7,
                   bf_b=0.639,
                   sf_s=0.150,
                   cf_c=0.198,
                   lambda_f=0.900),
        engine=Engines(max_thrust_sl=275800,
                       bypass_ratio=4.6,
                       zpos=0,
                       mount='wing',
                       num_motors=4,
                       cff1=2.88943,
                       cff2=-2.16939,
                       cff3=2.00708),
        limits=Limits(MTOW=396800),
        polar=Polar(CD0=0.021,
                    k=0.049,
                    e=0.816),
        landing_gear=LandingGear(x_ng=12,
                                 z_ng=3,
                                 nw_ng=1,
                                 x_mg=-1.2,
                                 z_mg=3,
                                 nw_mg=1),
        inertia=Inertia(Iy=44877574),
        coeffs=AerodynamicCoefficients(CL0=0.92,
                                       CLalpha=4.4,
                                       CLde=0.32,
                                       CLq=6.6,
                                       Cm0=0.0,
                                       Cmalpha=-1.00,
                                       Cmde=-1.30,
                                       Cmq=-1.30)
)
