import numpy as np
import openmdao.api as om

from toa.data.airplanes.airplanes import Airplanes


class FuelFlowComp(om.ExplicitComponent):
    """Computes fuel flow."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('condition', default='AEO', desc='Takeoff condition (AEO/OEI)')
        self.options.declare('airplane_data', types=Airplanes, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        airplane = self.options['airplane_data']

        self.add_input(name='thrust_ratio', shape=(nn,), desc='Thrust ratio', units=None)
        self.add_input(name='thrust', shape=(nn,), desc='Thrust at current elevation and speed', units='N')
        self.add_input(name='elevation', shape=(1,), desc='Runway elevation', units='m')

        self.add_output(name='dXdt:mass_fuel', val=np.zeros(nn),
                        desc='rate of aircraft mass change - negative when fuel is being depleted', units='kg/s')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zz = np.zeros(nn)

        self.declare_partials(of='dXdt:mass_fuel', wrt='thrust_ratio', rows=ar, cols=ar, method='fd', form='central',
                              step=1e-4)
        self.declare_partials(of='dXdt:mass_fuel', wrt='thrust', rows=ar, cols=ar, method='fd', form='central',
                              step=1e-4)
        self.declare_partials(of='dXdt:mass_fuel', wrt='elevation', rows=ar, cols=zz, method='fd', form='central',
                              step=1e-4)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']
        thrust_ratio = inputs['thrust_ratio']
        thrust = inputs['thrust']
        elevation = inputs['elevation']

        num_motors = airplane.engine.num_motors if self.options[
                                                       'condition'] == 'AEO' else airplane.engine.num_motors - 1

        ength = thrust / num_motors

        mass_fuel_eng = airplane.engine.cff3 * thrust_ratio ** 3 + airplane.engine.cff2 * thrust_ratio ** 2 + airplane.engine.cff1 * thrust_ratio + 6.7e-7 * (
                ength / 1000) * elevation

        outputs['dXdt:mass_fuel'] = mass_fuel_eng * num_motors
