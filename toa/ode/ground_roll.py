import numpy as np
import openmdao.api as om
from dymos.models.atmosphere import USatm1976Comp

from toa.data.airplanes.airplanes import Airplanes
from toa.models.aero.aerodynamics import AerodynamicsGroup
from toa.models.eom.ground_roll_eom import GroundRollEOM
from toa.models.propulsion.propulsion_group import PropulsionGroup
from toa.models.true_airspeed_comp import TrueAirspeedCompGroundRoll


class GroundRollODE(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('airplane_data', types=Airplanes, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']

        assumptions = self.add_subsystem(name='assumptions', subsys=om.IndepVarComp())
        assumptions.add_output('grav', val=9.80665, units='m/s**2', desc='Gravity acceleration')
        assumptions.add_output('alpha', val=0.0, units='rad', desc='Angle of Attack')

        self.connect('assumptions.alpha', 'alpha')

        self.add_subsystem(name='atmos', subsys=USatm1976Comp(num_nodes=1), promotes_inputs=['h'])

        self.add_subsystem(name='tas_comp', subsys=TrueAirspeedCompGroundRoll(num_nodes=nn))

        self.add_subsystem(name='aero', subsys=AerodynamicsGroup(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['alpha', 'de'], promotes_outputs=['L', 'D', 'M'])

        self.connect('assumptions.grav', 'aero.grav')
        self.connect('atmos.rho', 'aero.rho')

        self.add_subsystem(name='prop', subsys=PropulsionGroup(num_nodes=nn, airplane_data=airplane),
                           promotes_outputs=['thrust'])

        self.connect('tas_comp.tas', ['aero.tas', 'prop.tas'])
        self.connect('atmos.sos', 'prop.sos')
        self.connect('atmos.pres', 'prop.thrust_comp.p_amb')

        self.add_subsystem(name='ground_run_eom', subsys=GroundRollEOM(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['alpha', 'thrust'])

        self.connect('assumptions.grav', 'ground_run_eom.grav')
        self.connect('L', 'ground_run_eom.lift')
        self.connect('M', 'ground_run_eom.moment')
        self.connect('D', 'ground_run_eom.drag')