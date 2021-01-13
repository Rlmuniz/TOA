import jax
import numpy as np
import openmdao.api as om

from toa.data.airplanes.airplanes import Airplanes


class ThrustComp(om.ExplicitComponent):
    """Computes thrust ratio considering effects of altitude and speed."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('condition', default='AEO', desc='Takeoff condition (AEO/OEI)')
        self.options.declare('thrust_rating', default='takeoff', desc='Thrust rate (takeoff, idle)')
        self.options.declare('airplane_data', types=Airplanes, desc='Class containing all airplane data')
        self._init_gradients()

    def _init_gradients(self):
        """Generates function gradients for component."""
        argnums = (0, 1)
        in_axes = (0, None)

        thrust_ratio_grad = jax.vmap(jax.grad(self._compute_thrust_ratio, argnums), in_axes=in_axes)
        thrust_grad = jax.vmap(jax.grad(self._compute_thrust, argnums), in_axes=in_axes)

        self.grad_funcs = {
            'thrust_ratio': thrust_ratio_grad,
            'thrust': thrust_grad
        }

    def _compute_thrust_ratio(self, mach, p_amb):
        return self._compute(mach, p_amb)['thrust_ratio']

    def _compute_thrust(self, mach, p_amb):
        return self._compute(mach, p_amb)['thrust']

    def _compute(self, mach, p_amb):
        airplane = self.options['airplane_data']
        p_amb_sl = 101325.0

        press_ratio = p_amb / p_amb_sl

        A = - 0.4327 * press_ratio ** 2 + 1.3855 * press_ratio + 0.0472
        Z = 0.9106 * press_ratio ** 3 - 1.7736 * press_ratio ** 2 + 1.8697 * press_ratio
        X = 0.1377 * press_ratio ** 3 - 0.4374 * press_ratio ** 2 + 1.3003 * press_ratio

        multiplier = 1.0 if self.options['thrust_rating'] == 'takeoff' else 0.07
        num_motors = airplane.engine.num_motors if self.options[
                                                       'condition'] == 'AEO' else airplane.engine.num_motors - 1

        T_T0 = (A - airplane.engine.k1 * Z * mach + airplane.engine.k2 * X * mach ** 2) * multiplier
        return {
            'thrust_ratio': T_T0,
            'thrust': T_T0 * airplane.engine.max_thrust_sl * num_motors
        }

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input(name='p_amb', shape=(1,), desc='Atmospheric pressure', units='Pa')
        self.add_input(name='mach', shape=(nn,), desc='Mach number', units=None)

        # Outputs
        self.add_output(name='thrust_ratio', val=np.zeros(nn), desc='Thrust ratio at current altitude and speed',
                        units=None)
        self.add_output(name='thrust', val=np.zeros(nn), desc='Thrust at current altitude and speed', units='N')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zz = np.zeros(nn)

        self.declare_partials(of='thrust_ratio', wrt='p_amb', rows=ar, cols=zz)
        self.declare_partials(of='thrust_ratio', wrt='mach', rows=ar, cols=ar)
        self.declare_partials(of='thrust', wrt='p_amb', rows=ar, cols=zz)
        self.declare_partials(of='thrust', wrt='mach', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        p_amb, = inputs['p_amb']
        mach = inputs['mach']

        for out, res in self._compute(mach, p_amb).items():
            outputs[out] = res

    def compute_partials(self, inputs, partials, *kwargs):
        p_amb, = inputs['p_amb']
        mach = inputs['mach']

        wrt = 'mach', 'p_amb'
        args = mach, p_amb
        for of, grad_fun in self.grad_funcs.items():
            for var, res in zip(wrt, grad_fun(*args)):
                partials[of, var] = np.asarray(res, float)
