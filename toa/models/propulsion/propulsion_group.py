"""
Inputs:
 - True Airspeed
 - Speed of Sound
 - Ambient Pressure
 - Altitude (Field elevation + CG distance from field level)

 Outputs:
  - Thrust
  - dXdt:mass_fuel
"""

import openmdao.api as om

from toa.data import AirplaneData
from toa.models.propulsion.fuel_flow_comp import FuelFlowComp
from toa.models.propulsion.mach_comp import MachComp
from toa.models.propulsion.thrust_comp import ThrustComp


class PropulsionGroup(om.Group):
    """Computes the thrust and fuel flow considering the effect of speed and altitude."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('condition', default='AEO',
                             desc='Takeoff condition (AEO/OEI)')
        self.options.declare('thrust_rating', default='takeoff',
                             desc='Thrust rate (takeoff, idle)')
        self.options.declare('airplane_data', types=AirplaneData,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']
        condition = self.options['condition']
        thrust_rating = self.options['thrust_rating']

        self.add_subsystem(name='mach_comp', subsys=MachComp(num_nodes=nn),
                           promotes_inputs=['tas', 'sos'])

        self.connect('mach_comp.mach', 'thrust_comp.mach')

        self.add_subsystem(name='thrust_comp',
                           subsys=ThrustComp(num_nodes=nn, airplane_data=airplane,
                                             condition=condition,
                                             thrust_rating=thrust_rating),
                           promotes_outputs=['thrust'])

        self.connect('thrust_comp.thrust_ratio', 'fuel_flow.thrust_ratio')

        self.add_subsystem(name='fuel_flow',
                           subsys=FuelFlowComp(num_nodes=nn, airplane_data=airplane,
                                               condition=condition),
                           promotes_inputs=['thrust'],
                           promotes_outputs=['dXdt:mass_fuel'])
