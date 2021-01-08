import numpy as np
import openmdao.api as om
from dymos.models.atmosphere import USatm1976Comp

from toa.airplanes import AirplaneData
from toa.models.aero.aerodynamics import AerodynamicsGroup
from toa.models.eom.ground_roll_eom import GroundRollEOM
from toa.models.landing_gear.forces_comp import AllWheelsOnGroundReactionForces
from toa.models.propulsion.propulsion_group import PropulsionGroup
from toa.models.propulsion.thrust_comp import ThrustComp


class GroundRollODE(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']

        assumptions = self.add_subsystem('assumptions', subsys=om.IndepVarComp())
        assumptions.add_output(name='q', val=np.zeros(nn), desc='Pitch rate', units='rad/s')
        assumptions.add_output(name='alpha', val=np.zeros(nn), desc='Angle of attack', units='rad')
        self.connect('assumptions.q', 'q')
        self.connect('assumptions.alpha', 'alpha')

        self.add_subsystem(name='atmo', subsys=USatm1976Comp(num_nodes=nn))
        self.connect('atmo.rho', 'aero.rho')
        self.connect('atmo.pres', 'prop.p_amb')
        self.connect('atmo.sos', 'prop.sos')

        self.add_subsystem(name='aero', subsys=AerodynamicsGroup(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['alpha', 'de', 'v', 'q'], promotes_outputs=['L', 'D', 'M'])
        self.connect('aero.tas', 'prop.tas')
        self.connect('L', 'landing_gear.lift')
        self.connect('M', 'landing_gear.moment')
        self.connect('D', 'ground_run_eom.drag')

        self.add_subsystem(name='prop', subsys=PropulsionGroup(num_nodes=nn, airplane_data=airplane),
                           promotes_outputs=['thrust', 'dXdt:mass_fuel'])

        self.add_subsystem(name='landing_gear',
                           subsys=AllWheelsOnGroundReactionForces(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['mass', 'thrust'],
                           promotes_outputs=['mlg_reaction', 'nlg_reaction'])

        self.connect('mlg_reaction', 'ground_run_eom.mlg_reaction')
        self.connect('nlg_reaction', 'ground_run_eom.nlg_reaction')

        self.add_subsystem(name='ground_run_eom', subsys=GroundRollEOM(num_nodes=nn),
                           promotes_inputs=['mass', 'alpha', 'thrust'], promotes_outputs=['dXdt:v'])
