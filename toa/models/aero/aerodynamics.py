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

from toa.data import AirplaneData
from toa.models.aero.aero_forces_comp import AeroForcesComp
from toa.models.aero.coeff.aero_coef import AerodynamicsCoefficientsGroup
from toa.models.aero.dynamic_pressure_comp import DynamicPressureComp


class AerodynamicsGroup(om.Group):
    """Computes the lift and drag forces on the aircraft."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')
        self.options.declare('landing_gear', default=True, desc='Accounts landing gear drag')
        self.options.declare('AllWheelsOnGround', default=True)

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']
        landing_gear = self.options['landing_gear']
        all_wheels_on_ground = self.options['AllWheelsOnGround']

        self.add_subsystem(name='coeff_comp',
                           subsys=AerodynamicsCoefficientsGroup(num_nodes=nn,
                                                                airplane_data=airplane,
                                                                landing_gear=landing_gear,
                                                                AllWheelsOnGround=all_wheels_on_ground),
                           promotes_inputs=['alpha', 'de', 'flap_angle'],
                           promotes_outputs=['CL', 'CD', 'Cm'])

        self.add_subsystem(name='dyn_press',
                           subsys=DynamicPressureComp(num_nodes=nn),
                           promotes_inputs=['rho', 'tas'],
                           promotes_outputs=['qbar'])

        self.add_subsystem(name='forces_comp',
                           subsys=AeroForcesComp(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['CL', 'CD', 'Cm', 'qbar'],
                           promotes_outputs=['L', 'D', 'M'])
