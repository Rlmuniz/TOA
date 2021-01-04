import openmdao.api as om

from toa.airplanes import AirplaneData
from toa.models.dynamic_pressure_comp import DynamicPressureComp
from toa.models.ground_roll.true_airspeed_comp import TrueAirspeedComp
from toa.models.propulsion.thrust_comp import ThrustComp
from toa.models.rotation.aerodynamics import AerodynamicsGroup
from toa.models.rotation.rotation_eom import RotationEOM


class RotationODE(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')
        self.options.declare('num_motors', default=2, desc='Number of operating motors')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']
        n_motors = self.options['num_motors']

        assumptions = self.add_subsystem('assumptions', subsys=om.IndepVarComp())
        assumptions.add_output(name='grav', val=9.80665, desc='Gravity acceleration', units='m/s**2')
        assumptions.add_output(name='mu', val=0.002, desc='Friction coefficient', units=None)
        assumptions.add_output(name='rw_slope', val=0, desc='Runway rw_slope', units='rad')
        assumptions.add_output(name='vw', val=0, desc='Wind speed along the runway, defined as positive for a headwind', units='m/s')

        self.connect('assumptions.grav', 'rotation_eom.grav')
        self.connect('assumptions.mu', 'rotation_eom.mu')
        self.connect('assumptions.rw_slope', 'rotation_eom.rw_slope')
        self.connect('assumptions.vw', 'tas_comp.vw')

        self.add_subsystem(name='tas_comp', subsys=TrueAirspeedComp(num_nodes=nn), promotes_inputs=['v'])

        self.connect('tas_comp.tas', 'dyn_pressure.tas')
        self.connect('tas_comp.tas', 'aero.tas')

        self.add_subsystem(name='dyn_pressure', subsys=DynamicPressureComp(num_nodes=nn))

        self.connect('dyn_pressure.q', 'aero.q')

        self.add_subsystem(name='aero', subsys=AerodynamicsGroup(num_nodes=nn, airplane_data=airplane), promotes_inputs=['alpha', 'de', 'pitch_rate'], promotes_outputs=['L', 'D', 'M'])

        self.connect('L', 'rotation_eom.lift')
        self.connect('D', 'rotation_eom.drag')
        self.connect('M', 'rotation_eom.moment')

        self.add_subsystem(name='propulsion', subsys=ThrustComp(num_nodes=nn, num_motors=n_motors),
                           promotes_outputs=['T'])

        self.connect('T', 'rotation_eom.thrust')

        self.add_subsystem(name='rotation_eom', subsys=RotationEOM(num_nodes=nn, airplane_data=airplane), promotes_inputs=['mass', 'alpha'], promotes_outputs=['mlg_reaction'])