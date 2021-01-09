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
from toa.models.aero.aero_coef_comp import AeroCoeffComp
from toa.models.aero.aero_coef_comp import AeroCoeffCompInitialRun
from toa.models.aero.aero_forces_comp import AeroForcesComp
from toa.models.aero.dynamic_pressure_comp import DynamicPressureComp


class AerodynamicsGroup(om.Group):
    """Computes the lift and drag forces on the aircraft."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData,
                             desc='Class containing all airplane data')
        self.options.declare('phase', default='initial_run',
                             desc='Initial run, rotation, transition')

    def setup(self):
        nn = self.options['num_nodes']
        phase = self.options['phase']
        airplane = self.options['airplane_data']

        if phase == 'initial_run':
            self.add_subsystem(name='aero_coef_comp',
                               subsys=AeroCoeffCompInitialRun(num_nodes=nn,
                                                              airplane_data=airplane),
                               promotes_inputs=['alpha', 'de'],
                               promotes_outputs=['CL', 'CD', 'Cm'])
        else:
            self.add_subsystem(name='aero_coef_comp', subsys=AeroCoeffComp(num_nodes=nn,
                                                                           airplane_data=airplane),
                               promotes_inputs=['alpha', 'de', 'tas', 'q'],
                               promotes_outputs=['CL', 'CD', 'Cm'])

        self.add_subsystem(name='dyn_press', subsys=DynamicPressureComp(num_nodes=nn),
                           promotes_inputs=['rho', 'tas'], promotes_outputs=['qbar'])

        self.add_subsystem(name='aero_forces_comp',
                           subsys=AeroForcesComp(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['CL', 'CD', 'Cm', 'qbar'],
                           promotes_outputs=['L', 'D', 'M'])
