import numpy as np
import openmdao.api as om
from dymos.models.atmosphere import USatm1976Comp

from toa.airplanes import AirplaneData
from toa.models.aero.aerodynamics import AerodynamicsGroup
from toa.models.eom.rotation_eom import RotationEOM
from toa.models.propulsion.propulsion_group import PropulsionGroup


class RotationODE(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']

        assumptions = self.add_subsystem('assumptions', subsys=om.IndepVarComp())
        assumptions.add_output(name='alt', val=np.zeros(nn), desc='airplane CG elevation', units='m')
        assumptions.add_output(name='vw', val=0, desc='Wind speed along the runway, defined as positive for a headwind',
                               units='m/s')
        assumptions.add_output(name='grav', val=9.80665, desc='Gravity acceleration', units='m/s**2')
        assumptions.add_output(name='rw_slope', val=0.0, desc='Runway Slope', units='rad')

        self.connect('assumptions.alt', 'prop.alt')
        self.connect('assumptions.alt', 'atmo.h')
        self.connect('assumptions.vw', 'aero.vw')
        self.connect('assumptions.grav', 'rotation_eom.grav')
        self.connect('assumptions.rw_slope', 'rotation_eom.rw_slope')

        self.add_subsystem(name='atmo', subsys=USatm1976Comp(num_nodes=nn))
        self.connect('atmo.rho', 'aero.rho')
        self.connect('atmo.pres', 'prop.p_amb')
        self.connect('atmo.sos', 'prop.sos')

        self.add_subsystem(name='aero', subsys=AerodynamicsGroup(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['alpha', 'de', 'v', 'q'], promotes_outputs=['L', 'D', 'M'])
        self.connect('aero.tas', 'prop.tas')
        self.connect('L', 'rotation_eom.lift')
        self.connect('D', 'rotation_eom.drag')
        self.connect('M', 'rotation_eom.moment')

        self.add_subsystem(name='prop', subsys=PropulsionGroup(num_nodes=nn, airplane_data=airplane),
                           promotes_outputs=['thrust', 'dXdt:mass_fuel'])

        self.add_subsystem(name='rotation_eom', subsys=RotationEOM(num_nodes=nn, airplane_data=airplane),
                           promotes_inputs=['mass', 'alpha', 'thrust'], promotes_outputs=['dXdt:v', 'dXdt:q', 'mlg_reaction'])
