"""
Inputs:
 - v
 - vw
 - alpha
 - de
 - q
 - rho

Outputs:
  - CL, CD, CM
  - L, D, M
"""

import openmdao.api as om

from toa.airplanes import AirplaneData
from toa.models.aero.aero_coef_comp import AeroCoeffComp
from toa.models.aero.aero_forces_comp import AeroForcesComp
from toa.models.aero.dynamic_pressure_comp import DynamicPressureComp
from toa.models.aero.true_airspeed_comp import TrueAirspeedComp


class AerodynamicsGroup(om.Group):
    """Computes the lift and drag forces on the aircraft."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']

        self.add_subsystem(name='tas_comp', subsys=TrueAirspeedComp(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['v', 'vw'])
        self.connect('tas_comp.tas', 'aero_coef_comp.tas')
        self.connect('tas_comp.tas', 'dyn_press.tas')

        self.add_subsystem(name='aero_coef_comp', subsys=AeroCoeffComp(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['alpha', 'de', 'q'],
                           promotes_outputs=['CL', 'CD', 'Cm'])

        self.add_subsystem(name='dyn_press', subsys=DynamicPressureComp(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['rho'])
        self.connect('dyn_press.qbar', 'aero_forces_comp.qbar')

        self.add_subsystem(name='aero_forces_comp', subsys=AeroForcesComp(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['CL', 'CD', 'Cm'], promotes_outputs=['L', 'D', 'M'])
