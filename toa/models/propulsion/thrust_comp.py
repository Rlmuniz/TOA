import jax
import numpy as np
import openmdao.api as om

from toa.data import Airplane


class ThrustComp(om.ExplicitComponent):
    """Computes thrust ratio considering effects of altitude and speed."""

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

        # Inputs
        self.add_input(name='p_amb', shape=(1,), desc='Atmospheric pressure',
                       units='Pa')
        self.add_input(name='mach', shape=(nn,), desc='Mach number', units=None)

        # Outputs
        self.add_output(name='thrust_ratio', val=np.zeros(nn),
                        desc='Thrust ratio at current altitude and speed',
                        units=None)
        self.add_output(name='thrust', val=np.zeros(nn),
                        desc='Thrust at current altitude and speed', units='N')

        self.declare_partials(of='thrust_ratio', wrt=['*'], method='fd')
        self.declare_partials(of='thrust', wrt=['*'], method='fd')

    def compute(self, inputs, outputs, **kwargs):
        p_amb, = inputs['p_amb']
        mach = inputs['mach']
        ap = self.options['airplane']

        bpr = ap.engine.bypass_ratio
        p_amb_sl = 101325.0

        press_ratio = p_amb / p_amb_sl

        A = - 0.4327 * press_ratio ** 2 + 1.3855 * press_ratio + 0.0472
        Z = 0.9106 * press_ratio ** 3 - 1.7736 * press_ratio ** 2 + 1.8697 * press_ratio
        X = 0.1377 * press_ratio ** 3 - 0.4374 * press_ratio ** 2 + 1.3003 * press_ratio

        multiplier = 1.0 if self.options['throttle'] == 'takeoff' else 0.07

        if self.options['condition'] == 'AEO':
            num_motors = ap.engine.num_motors
        else:
            num_motors = ap.engine.num_motors - 1

        G0 = 0.0606 * bpr + 0.6337
        k1 = 0.377 * (1 + bpr) / np.sqrt((1 + 0.82 * bpr) * G0)
        k2 = 0.23 + 0.19 * np.sqrt(bpr)

        T_T0 = (A - k1 * Z * mach + k2 * X * mach ** 2) * multiplier

        outputs['thrust_ratio'] = T_T0
        outputs['thrust'] = T_T0 * ap.engine.max_thrust_sl * num_motors
