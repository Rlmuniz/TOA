import openmdao.api as om
from dymos.models.atmosphere import USatm1976Comp

from toa.data import Airplane
from toa.models.aero.aerodynamics import AerodynamicsGroup
from toa.models.eom.rotation_eom import RotationEOM
from toa.models.propulsion.propulsion_group import PropulsionGroup
from toa.models.true_airspeed_comp import TrueAirspeedCompGroundRoll


class RotationODE(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane']

        assumptions = self.add_subsystem(name='assumptions', subsys=om.IndepVarComp())
        assumptions.add_output('grav', val=9.80665, units='m/s**2',
                               desc='Gravity acceleration')

        self.add_subsystem(name='atmos', subsys=USatm1976Comp(num_nodes=1),
                           promotes_inputs=['h'])

        self.add_subsystem(name='tas_comp',
                           subsys=TrueAirspeedCompGroundRoll(num_nodes=nn))

        self.add_subsystem(name='aero', subsys=AerodynamicsGroup(num_nodes=nn,
                                                                 airplane=airplane,
                                                                 AllWheelsOnGround=False),
                           promotes_inputs=['alpha', 'de'])

        self.connect('assumptions.grav', 'aero.grav')
        self.connect('atmos.rho', 'aero.rho')

        self.add_subsystem(name='prop',
                           subsys=PropulsionGroup(num_nodes=nn, airplane=airplane))

        self.connect('atmos.sos', 'prop.sos')
        self.connect('atmos.pres', 'prop.thrust_comp.p_amb')
        self.connect('tas_comp.tas', ['aero.tas', 'prop.tas'])

        self.add_subsystem(name='rotation_eom',
                           subsys=RotationEOM(num_nodes=nn, airplane=airplane),
                           promotes_inputs=['alpha'])

        self.connect('assumptions.grav', 'rotation_eom.grav')
        self.connect('aero.L', 'rotation_eom.lift')
        self.connect('aero.D', 'rotation_eom.drag')
        self.connect('aero.M', 'rotation_eom.moment')
        self.connect('prop.thrust', 'rotation_eom.thrust')
