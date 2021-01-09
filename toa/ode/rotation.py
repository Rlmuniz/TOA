import openmdao.api as om

from toa.data import AirplaneData
from toa.models.aero.aerodynamics import AerodynamicsGroup
from toa.models.eom.rotation_eom import RotationEOM
from toa.models.propulsion.propulsion_group import PropulsionGroup
from toa.models.true_airspeed_comp import TrueAirspeedCompGroundRoll


class RotationODE(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']

        self.add_subsystem(name='tas_comp', subsys=TrueAirspeedCompGroundRoll(num_nodes=nn))
        self.connect('tas_comp.tas', ['aero.tas', 'prop.tas'])

        self.add_subsystem(name='aero', subsys=AerodynamicsGroup(num_nodes=nn, airplane_data=airplane, phase='rotation'),
                           promotes_inputs=['alpha', 'de'], promotes_outputs=['L', 'D', 'M'])

        self.connect('L', 'rotation_eom.lift')
        self.connect('D', 'rotation_eom.drag')
        self.connect('M', 'rotation_eom.moment')

        self.add_subsystem(name='prop', subsys=PropulsionGroup(num_nodes=nn, airplane_data=airplane),
                           promotes_outputs=['thrust'])

        self.add_subsystem(name='rotation_eom', subsys=RotationEOM(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['alpha', 'thrust'])
