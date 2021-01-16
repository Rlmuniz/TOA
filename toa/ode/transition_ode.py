import openmdao.api as om
from dymos.models.atmosphere import USatm1976Comp

from toa.data import Airplane
from toa.models.aero.aerodynamics import AerodynamicsGroup
from toa.models.alpha_comp import AlphaComp
from toa.models.eom.transition_oem import TransitionOEM
from toa.models.propulsion.propulsion_group import PropulsionGroup


class TransitionODE(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')
        self.options.declare('condition', default='AEO',
                             desc='Takeoff condition (AEO/OEI)')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane']
        condition = self.options['condition']

        assumptions = self.add_subsystem(name='assumptions', subsys=om.IndepVarComp())
        assumptions.add_output('grav', val=9.80665, units='m/s**2',
                               desc='Gravity acceleration')

        self.add_subsystem(name='atmos', subsys=USatm1976Comp(num_nodes=1),
                           promotes_inputs=['h'])

        self.add_subsystem(name='tas_comp',
                           subsys=TrueAirspeedCompTransition(num_nodes=nn),
                           promotes_inputs=['V', 'Vw', 'gam'])

        self.add_subsystem(name='alpha_comp', subsys=AlphaComp(num_nodes=nn),
                           promotes_inputs=['theta'])

        self.connect('tas_comp.corr_gam', 'alpha_comp.gam')

        self.add_subsystem(name='aero', subsys=AerodynamicsGroup(num_nodes=nn,
                                                                 airplane=airplane,
                                                                 AllWheelsOnGround=False),
                           promotes_inputs=['de', 'q', 'mass'])

        self.connect('assumptions.grav', 'aero.grav')
        self.connect('atmos.rho', 'aero.rho')
        self.connect('tas_comp.tas', 'aero.tas')
        self.connect('alpha_comp.alpha', 'aero.alpha')

        self.add_subsystem(name='prop',
                           subsys=PropulsionGroup(num_nodes=nn, airplane=airplane,
                                                  condition=condition))

        self.connect('atmos.sos', 'prop.sos')
        self.connect('atmos.pres', 'prop.p_amb')
        self.connect('tas_comp.tas', 'prop.tas')

        self.add_subsystem(name='transition_eom',
                           subsys=TransitionOEM(num_nodes=nn, airplane=airplane),
                           promotes_inputs=['q', 'mass'])

        self.connect('prop.thrust', 'transition_eom.thrust')
        self.connect('aero.L', 'transition_oem.lift')
        self.connect('aero.D', 'transition_oem.drag')
        self.connect('aero.M', 'transition_oem.moment')
        self.connect('tas_comp.tas', 'transition_eom.V')
        self.connect('tas_comp.corr_gam', 'transition_eom.gam')
        self.connect('assumptions.grav', 'transition_eom.grav')
        self.connect('alpha_comp.alpha', 'transition_oem.alpha')
