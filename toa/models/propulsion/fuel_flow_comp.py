import numpy as np
import openmdao.api as om

from toa.data import Airplane


class FuelFlowComp(om.ExplicitComponent):
    """Computes fuel flow."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('condition', default='AEO',
                             desc='Takeoff condition (AEO/OEI)')
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='thrust_ratio', shape=(nn,), desc='Thrust ratio',
                       units=None)
        self.add_input(name='thrust', shape=(nn,),
                       desc='Thrust at current elevation and speed', units='N')
        self.add_input(name='elevation', shape=(1,), desc='Runway elevation', units='m')

        self.add_output(name='m_dot', val=np.zeros(nn),
                        desc='rate of aircraft mass change - negative when fuel is being depleted',
                        units='kg/s')

        self.declare_partials(of='m_dot', wrt=['*'], method='fd')

    def compute(self, inputs, outputs, **kwargs):
        thrust_ratio = inputs['thrust_ratio']
        thrust = inputs['thrust']
        elevation = inputs['elevation']

        ap = self.options['airplane']

        if self.options['condition'] == 'AEO':
            num_motors = ap.engine.num_motors
        else:
            num_motors = ap.engine.num_motors - 1

        ength = thrust / num_motors

        mass_fuel_eng = (
                ap.engine.cff3 * thrust_ratio ** 3
                + ap.engine.cff2 * thrust_ratio ** 2
                + ap.engine.cff1 * thrust_ratio + 6.7e-7
                * (ength / 1000) * elevation
        )

        outputs['m_dot'] = -1.0 * mass_fuel_eng * num_motors
