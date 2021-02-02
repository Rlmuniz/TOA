import openmdao.api as om
from dymos.models.atmosphere import USatm1976Comp

from toa.data import Airplane
from toa.models.aero.aerodynamics import AerodynamicsGroup
from toa.models.alpha_comp import AlphaComp
from toa.models.eom.transition_oem import TransitionOEM
from toa.models.main_landing_gear_pos import MainLandingGearPosComp
from toa.models.propulsion.propulsion_group import PropulsionGroup
from toa.models.true_airspeed_comp import TrueAirspeedComp
from toa.models.v_vs_comp import VVstallRatioComp


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
                           promotes_inputs=[('h', 'elevation')])

        self.add_subsystem(name='tas_comp',
                           subsys=TrueAirspeedComp(num_nodes=nn),
                           promotes_inputs=['V', 'Vw', 'gam'])

        self.add_subsystem(name='alpha_comp', subsys=AlphaComp(num_nodes=nn),
                           promotes_inputs=['theta', 'gam'])

        self.add_subsystem(name='aero', subsys=AerodynamicsGroup(num_nodes=nn,
                                                                 airplane=airplane,
                                                                 AllWheelsOnGround=False),
                           promotes_inputs=['q', 'mass'])

        self.connect('assumptions.grav', 'aero.grav')
        self.connect('atmos.rho', 'aero.rho')
        self.connect('tas_comp.tas', 'aero.tas')
        self.connect('alpha_comp.alpha', 'aero.alpha')

        self.add_subsystem(name='prop',
                           subsys=PropulsionGroup(num_nodes=nn, airplane=airplane,
                                                  condition=condition),
                           promotes_inputs=['elevation'])

        self.connect('atmos.sos', 'prop.sos')
        self.connect('atmos.pres', 'prop.p_amb')
        self.connect('tas_comp.tas', 'prop.tas')

        self.add_subsystem(name='transition_eom',
                           subsys=TransitionOEM(num_nodes=nn, airplane=airplane),
                           promotes_inputs=['q', 'mass', 'V', 'Vw', 'gam'])

        self.connect('prop.thrust', 'transition_eom.thrust')
        self.connect('aero.L', 'transition_eom.lift')
        self.connect('aero.D', 'transition_eom.drag')
        self.connect('aero.M', 'transition_eom.moment')
        self.connect('assumptions.grav', 'transition_eom.grav')
        self.connect('alpha_comp.alpha', 'transition_eom.alpha')

        self.add_subsystem(name='mlg_pos',
                           subsys=MainLandingGearPosComp(num_nodes=nn,
                                                         airplane=airplane),
                           promotes_inputs=['theta'])

        self.add_subsystem(name='v_vs_comp', subsys=VVstallRatioComp(num_nodes=nn, airplane=airplane),
                           promotes_inputs=['mass', 'V'])

        self.connect('aero.CLmax', 'v_vs_comp.CLmax')
        self.connect('assumptions.grav', 'v_vs_comp.grav')
        self.connect('atmos.rho', 'v_vs_comp.rho')

        self.set_input_defaults('elevation', val=0.0, units='m')
