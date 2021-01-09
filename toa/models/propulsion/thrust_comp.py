import numpy as np
import openmdao.api as om

from toa.airplanes import AirplaneData


class ThrustComp(om.ExplicitComponent):
    """Computes thrust ratio considering effects of altitude and speed."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('condition', default='AEO', desc='Takeoff condition (AEO/OEI)')
        self.options.declare('thrust_rating', default='takeoff', desc='Thrust rate (takeoff, idle)')
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.p_amb_sl = 101325.0  # Pa
        # Inputs
        self.add_input(name='p_amb', shape=(1,), desc='Atmospheric pressure', units='Pa')
        self.add_input(name='mach', shape=(nn,), desc='Mach number', units=None)

        # Outputs
        self.add_output(name='thrust_ratio', val=np.zeros(nn), desc='Thrust ratio at current altitude and speed', units=None)
        self.add_output(name='thrust', val=np.zeros(nn), desc='Thrust at current altitude and speed', units='N')

    def setup_partials(self):
        self.declare_partials(of='thrust_ratio', wrt='p_amb', method='fd', form='central', step=1e-4)
        self.declare_partials(of='thrust_ratio', wrt='mach', method='fd', form='central', step=1e-4)
        self.declare_partials(of='thrust', wrt='p_amb', method='fd', form='central', step=1e-4)
        self.declare_partials(of='thrust', wrt='mach', method='fd', form='central', step=1e-4)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']
        p_amb = inputs['p_amb']
        mach = inputs['mach']

        press_ratio = p_amb / self.p_amb_sl
        A = - 0.4327 * press_ratio ** 2 + 1.3855 * press_ratio + 0.0472
        Z = 0.9106 * press_ratio ** 3 - 1.7736 * press_ratio ** 2 + 1.8697 * press_ratio
        X = 0.1377 * press_ratio ** 3 - 0.4374 * press_ratio ** 2 + 1.3003 * press_ratio

        multiplier = 1.0 if self.options['thrust_rating'] == 'takeoff' else 0.07
        num_motors = airplane.num_motors if self.options['condition'] == 'AEO' else airplane.num_motors - 1

        outputs['thrust_ratio'] = (A - airplane.k1 * Z * mach + airplane.k2 * X * mach ** 2) * multiplier
        outputs['thrust'] = outputs['thrust_ratio'] * airplane.max_thrust_sl * num_motors
