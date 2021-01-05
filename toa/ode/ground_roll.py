import openmdao.api as om

from toa.airplanes import AirplaneData
from toa.models.dynamic_pressure_comp import DynamicPressureComp
from toa.models.aero.aerodynamics import AerodynamicsGroup
from toa.models.ground_roll.ground_roll_eom import GroundRollEOM
from toa.models.landing_gear.forces import AllWheelsOnGroundReactionForces
from toa.models.aero.true_airspeed_comp import TrueAirspeedComp
from toa.models.propulsion.thrust_comp import ThrustComp


class GroundRollODE(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')
        self.options.declare('num_motors', default=2, desc='Number of operating motors')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']
        n_motors = self.options['num_motors']

        assumptions = self.add_subsystem('assumptions', subsys=om.IndepVarComp())
        assumptions.add_output(name='rho', val=1.1, desc='Density', units='kg/m**3')
        assumptions.add_output(name='grav', val=9.80665, desc='Gravity acceleration', units='m/s**2')
        assumptions.add_output(name='mu', val=0.002, desc='Friction coefficient', units=None)
        assumptions.add_output(name='rw_slope', val=0, desc='Runway rw_slope', units='rad')
        assumptions.add_output(name='vw', val=0, desc='Wind speed along the runway, defined as positive for a headwind', units='m/s')
        assumptions.add_output(name='alpha', val=0, desc='angle of attack', units='rad')

        self.connect('assumptions.rho', 'dyn_pressure.rho')
        self.connect('assumptions.grav', ('landing_gear.grav', 'ground_run_eom.grav'))
        self.connect('assumptions.mu', ('landing_gear.mu', 'ground_run_eom.mu'))
        self.connect('assumptions.rw_slope', ('landing_gear.rw_slope', 'ground_run_eom.rw_slope'))
        self.connect('assumptions.vw', 'tas_comp.vw')
        self.connect('assumptions.alpha', ('aero.alpha', 'ground_run_eom.alpha'))

        self.add_subsystem(name='tas_comp', subsys=TrueAirspeedComp(num_nodes=nn), promotes_inputs=['v'])

        self.connect('tas_comp.tas', 'dyn_pressure.tas')

        self.add_subsystem(name='dyn_pressure', subsys=DynamicPressureComp(num_nodes=nn))

        self.connect('dyn_pressure.q', 'aero.q')

        self.add_subsystem(name='aero', subsys=AerodynamicsGroup(num_nodes=nn, airplane_data=airplane), promotes_inputs=['alpha', 'de'], promotes_outputs=['L', 'D', 'M'])

        self.connect('L', 'landing_gear.lift')
        self.connect('M', 'landing_gear.moment')
        self.connect('D', 'ground_run_eom.drag')

        self.add_subsystem(name='propulsion', subsys=ThrustComp(num_nodes=nn, num_motors=n_motors),
                           promotes_outputs=['T'])

        self.connect('T', ('landing_gear.thrust', 'ground_run_eom.thrust'))

        self.add_subsystem(name='landing_gear',
                           subsys=AllWheelsOnGroundReactionForces(num_nodes=nn, airplane_data=airplane), promotes_inputs=['mass'],
                           promotes_outputs=['mlg_reaction', 'nlg_reaction'])

        self.connect('mlg_reaction', 'ground_run_eom.mlg_reaction')
        self.connect('nlg_reaction', 'ground_run_eom.nlg_reaction')

        self.add_subsystem(name='ground_run_eom', subsys=GroundRollEOM(num_nodes=nn), promotes_inputs=['mass', 'alpha'])
