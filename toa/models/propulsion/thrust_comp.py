import jax
import numpy as np
import openmdao.api as om

from toa.data import Airplane
from toa.data import get_airplane_data


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
        self.add_input(name='p_amb', val=np.zeros(nn), desc='Atmospheric pressure',
                       units='Pa')
        self.add_input(name='mach', val=np.zeros(nn), desc='Mach number', units=None)

        # Outputs
        self.add_output(name='thrust_ratio', val=np.zeros(nn),
                        desc='Thrust ratio at current altitude and speed',
                        units=None)
        self.add_output(name='thrust', val=np.zeros(nn),
                        desc='Thrust at current altitude and speed', units='N')

        self.declare_partials(of='thrust_ratio', wrt='p_amb', method='fd')
        self.declare_partials(of='thrust', wrt=['*'], method='fd')

    def compute(self, inputs, outputs, **kwargs):
        p_amb = inputs['p_amb']
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

    def compute_partials(self, inputs, partials, **kwargs):
        ap = self.options['airplane']
        p_amb = inputs['p_amb']
        mach = inputs['mach']

        p_amb_sl = 101325.0
        press_ratio = p_amb / p_amb_sl

        multiplier = 1.0 if self.options['throttle'] == 'takeoff' else 0.07

        if self.options['condition'] == 'AEO':
            num_motors = ap.engine.num_motors
        else:
            num_motors = ap.engine.num_motors - 1

        bpr = ap.engine.bypass_ratio
        max_thrust_sl = ap.engine.max_thrust_sl

        G0 = 0.0606 * bpr + 0.6337
        k1 = 0.377 * (1 + bpr) / np.sqrt((1 + 0.82 * bpr) * G0)
        k2 = 0.23 + 0.19 * np.sqrt(bpr)

        partials['thrust_ratio', 'p_amb'] = -multiplier * (mach * (
                k1 * (2.7318 * p_amb ** 2 - 3.5472 * p_amb * p_amb_sl + 1.8697 * p_amb_sl ** 2) - k2 * mach * (
                0.4131 * p_amb ** 2 - 0.8748 * p_amb * p_amb_sl + 1.3003 * p_amb_sl ** 2))
                                                           + 0.8654 * p_amb * p_amb_sl
                                                           - 1.3855 * p_amb_sl ** 2) / p_amb_sl ** 3
        partials['thrust_ratio', 'mach'] = -multiplier * p_amb * (
                k1 * (0.9106 * p_amb ** 2 - 1.7736 * p_amb * p_amb_sl + 1.8697 * p_amb_sl ** 2) - 2 * k2 * mach * (
                0.1377 * p_amb ** 2 - 0.4374 * p_amb * p_amb_sl + 1.3003 * p_amb_sl ** 2)) / p_amb_sl ** 3

        partials['thrust', 'p_amb'] = max_thrust_sl * num_motors * partials['thrust_ratio', 'p_amb']
        partials['thrust', 'mach'] = max_thrust_sl * num_motors * partials['thrust_ratio', 'mach']


if __name__ == '__main__':
    prob = om.Problem()
    airplane = get_airplane_data('b734')
    num_nodes = 1
    prob.model.add_subsystem('comp', ThrustComp(num_nodes=1, airplane=airplane))

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    prob.check_partials(compact_print=True, show_only_incorrect=True)
