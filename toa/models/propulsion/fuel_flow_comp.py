import numpy as np
import openmdao.api as om

from toa.airplanes import AirplaneData


class FuelFlowComp(om.ExplicitComponent):
    """Computes fuel flow."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('condition', default='AEO',
                             desc='Takeoff condition (AEO/OEI)')
        self.options.declare('airplane_data', types=AirplaneData,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']

        self.add_input(name='thrust_ratio', shape=(nn,), desc='Thrust ratio',
                       units=None)
        self.add_input(name='thrust', shape=(nn,),
                       desc='Thrust at current elevation and speed', units='N')
        self.add_input(name='elevation', shape=(1,), desc='Runway elevation', units='m')

        self.add_output(name='dXdt:mass_fuel', val=np.zeros(nn),
                        desc='rate of aircraft mass change - negative when fuel is being depleted',
                        units='kg/s')

    def setup_partials(self):
        self.declare_partials(of='dXdt:mass_fuel', wrt='thrust_ratio')
        self.declare_partials(of='dXdt:mass_fuel', wrt='thrust')
        self.declare_partials(of='dXdt:mass_fuel', wrt='elevation')

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']
        thrust_ratio = inputs['thrust_ratio']
        thrust = inputs['thrust']
        elevation = inputs['elevation']

        num_motors = airplane.num_motors if self.options[
                                                'condition'] == 'AEO' else airplane.num_motors - 1

        ength = thrust / num_motors

        mass_fuel_eng = airplane.cff3 * thrust_ratio ** 3 + airplane.cff2 * thrust_ratio ** 2 + airplane.cff1 * thrust_ratio + 6.7e-7 * (
                ength / 1000) * elevation

        outputs['dXdt:mass_fuel'] = mass_fuel_eng * num_motors

    def compute_partials(self, inputs, partials, **kwargs):
        airplane = self.options['airplane_data']
        thrust_ratio = inputs['thrust_ratio']
        thrust = inputs['thrust']
        elevation = inputs['elevation']

        num_motors = airplane.num_motors if self.options[
                                                'condition'] == 'AEO' else airplane.num_motors - 1

        partials['dXdt:mass_fuel', 'thrust_ratio'] = num_motors * (
                airplane.cff1 + 2 * airplane.cff2 * thrust_ratio + 3 * airplane.cff3 * thrust_ratio ** 2)
        partials['dXdt:mass_fuel', 'thrust'] = 6.7e-10 * elevation
        partials['dXdt:mass_fuel', 'elevation'] = 6.7e-10 * thrust
