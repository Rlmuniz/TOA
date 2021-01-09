import numpy as np
import openmdao.api as om
from dymos.models.atmosphere import USatm1976Comp

from toa.airplanes import AirplaneData
from toa.models.aero.aerodynamics import AerodynamicsGroup
from toa.models.eom.ground_roll_eom import GroundRollEOM
from toa.models.landing_gear.forces_comp import AllWheelsOnGroundReactionForces
from toa.models.propulsion.propulsion_group import PropulsionGroup
from toa.models.true_airspeed_comp import TrueAirspeedCompGroundRoll


class GroundRollODE(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']

        self.add_subsystem(name='tas_comp', subsys=TrueAirspeedCompGroundRoll(num_nodes=nn))
        self.connect('tas_comp.tas', ['aero.tas', 'prop.tas'])

        self.add_subsystem(name='aero', subsys=AerodynamicsGroup(num_nodes=nn, airplane_data=airplane, phase='initial_run'),
                           promotes_inputs=['alpha', 'de'], promotes_outputs=['L', 'D', 'M'])

        self.connect('L', 'ground_run_eom.lift')
        self.connect('M', 'ground_run_eom.moment')
        self.connect('D', 'ground_run_eom.drag')

        self.add_subsystem(name='prop', subsys=PropulsionGroup(num_nodes=nn, airplane_data=airplane),
                           promotes_outputs=['thrust'])

        self.add_subsystem(name='ground_run_eom', subsys=GroundRollEOM(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['alpha', 'thrust'])
