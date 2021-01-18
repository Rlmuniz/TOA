import openmdao.api as om

from toa.data import Airplane
from toa.models.propulsion.fuel_flow_comp import FuelFlowComp
from toa.models.propulsion.mach_comp import MachComp
from toa.models.propulsion.thrust_comp import ThrustComp


class PropulsionGroup(om.Group):
    """Computes the thrust and fuel flow considering the effect of speed and altitude."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('condition', default='AEO',
                             desc='Takeoff condition (AEO/OEI)')
        self.options.declare('throttle', default='takeoff',
                             desc='Thrust rate (takeoff, idle)')
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane']
        condition = self.options['condition']
        throttle = self.options['throttle']

        self.add_subsystem(name='mach_comp', subsys=MachComp(num_nodes=nn),
                           promotes_inputs=['tas', 'sos'])

        self.add_subsystem(name='thrust_comp',
                           subsys=ThrustComp(num_nodes=nn, airplane=airplane,
                                             condition=condition,
                                             throttle=throttle),
                           promotes_inputs=['p_amb'],
                           promotes_outputs=['thrust'])

        self.connect('mach_comp.mach', 'thrust_comp.mach')

        self.add_subsystem(name='fuel_flow',
                           subsys=FuelFlowComp(num_nodes=nn, airplane=airplane,
                                               condition=condition),
                           promotes_inputs=['thrust', 'elevation'],
                           promotes_outputs=['m_dot'])

        self.connect('thrust_comp.thrust_ratio', 'fuel_flow.thrust_ratio')
