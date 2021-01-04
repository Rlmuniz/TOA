import openmdao.api as om

from toa.airplanes import AirplaneData
from toa.models.ground_roll.aero.aero_coef_comp import AeroCoeffComp
from toa.models.ground_roll.aero.aero_forces_comp import AeroForcesComp


class AerodynamicsGroup(om.Group):
    """Computes the lift and drag forces on the aircraft."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']

        self.add_subsystem(name='aero_coef_comp', subsys=AeroCoeffComp(num_nodes=nn, airplane_data=airplane), promotes_inputs=['alpha', 'de'],
                           promotes_outputs=['CL', 'CD', 'Cm'])

        self.add_subsystem(name='aero_forces_comp', subsys=AeroForcesComp(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['q', 'CL', 'CD', 'Cm'], promotes_outputs=['L', 'D', 'M'])
