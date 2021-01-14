import jax
import numpy as np
import jax.numpy as jnp
import openmdao.api as om

from toa.data.airplanes.airplanes import Airplanes


class RotationEOM(om.ExplicitComponent):
    """Models the rotation phase (2 DoF) in the takeoff run."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=Airplanes,
                             desc='Class containing all airplane data')
        self._init_gradients()

    def _init_gradients(self):
        """Generates function gradients for component."""
        argnums = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        in_axes = (0, 0, 0, 0, 0, 0, 0, 0, None, None)

        dXdt_v_grad = jax.vmap(jax.grad(self._compute_dXdt_v, argnums), in_axes=in_axes)
        dXdt_q_grad = jax.vmap(jax.grad(self._compute_dXdt_q, argnums), in_axes=in_axes)
        rf_mainwheel_grad = jax.vmap(jax.grad(self._compute_rf_mainwheel, argnums), in_axes=in_axes)

        self.grad_funcs = {
            'dXdt:v': dXdt_v_grad,
            'dXdt:q': dXdt_q_grad,
            'rf_mainwheel': rf_mainwheel_grad
        }

    def _compute_dXdt_v(self, thrust, lift, drag, moment, V, mass, alpha, q, grav, rw_slope):
        return self._compute(thrust, lift, drag, moment, V, mass, alpha, q, grav, rw_slope)['dXdt:v']

    def _compute_dXdt_q(self, thrust, lift, drag, moment, V, mass, alpha, q, grav, rw_slope):
        return self._compute(thrust, lift, drag, moment, V, mass, alpha, q, grav, rw_slope)['dXdt:q']

    def _compute_rf_mainwheel(self, thrust, lift, drag, moment, V, mass, alpha, q, grav, rw_slope):
        return self._compute(thrust, lift, drag, moment, V, mass, alpha, q, grav, rw_slope)['rf_mainwheel']

    def _compute(self, thrust, lift, drag, moment, V, mass, alpha, q, grav, rw_slope):
        airplane = self.options['airplane_data']
        mu = 0.002

        weight = mass * grav

        rf_mainwheel = weight * jnp.cos(rw_slope) - lift
        f_rr = mu * rf_mainwheel
        m_mainwheel = airplane.landing_gear.x_mg * rf_mainwheel

        return {
            'dXdt:v': (thrust * jnp.cos(alpha) - drag - f_rr - weight * jnp.sin(rw_slope)) / mass,
            'dXdt:x': V,
            'dXdt:q': (moment + m_mainwheel) / airplane.inertia.Iy,
            'dXdt:alpha': q,
            'rf_mainwheel': rf_mainwheel
        }

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='thrust', val=np.zeros(nn), desc='Engine total thrust', units='N')
        self.add_input(name='lift', val=np.zeros(nn), desc='Lift', units='N')
        self.add_input(name='drag', val=np.zeros(nn), desc='Drag force', units='N')
        self.add_input(name='moment', val=np.zeros(nn), desc='Aerodynamic moment', units='N*m')
        self.add_input(name='V', val=np.zeros(nn), desc='Body x axis velocity', units='m/s')
        self.add_input(name='mass', val=np.zeros(nn), desc='Airplane mass', units='kg')
        self.add_input(name='alpha', val=np.zeros(nn), desc='Angle of attack', units='rad')
        self.add_input('q', val=np.zeros(nn), desc='Pitch rate', units='rad/s')
        self.add_input(name='rw_slope', val=0.0, desc='Runway slope', units='rad')
        self.add_input(name='grav', val=0.0, desc='Gravity acceleration', units='m/s**2')

        self.add_output('dXdt:x', val=np.zeros(nn), desc='Derivative of position', units='m/s')
        self.add_output(name='dXdt:v', val=np.zeros(nn), desc="Body x axis acceleration", units='m/s**2')
        self.add_output(name='dXdt:alpha', val=np.zeros(nn), desc="Alpha derivative", units='rad/s')
        self.add_output(name='dXdt:q', val=np.zeros(nn), desc="Pitch rate derivative", units='rad/s**2')
        self.add_output(name='rf_mainwheel', val=np.zeros(nn), desc='Main wheel reaction force', units='N')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zz = np.zeros(nn)

        self.declare_partials(of='dXdt:v', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='moment', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='V', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='q', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='dXdt:v', wrt='rw_slope', rows=ar, cols=zz)

        self.declare_partials(of='dXdt:x', wrt='V', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='dXdt:alpha', wrt='q', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='dXdt:q', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:q', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:q', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:q', wrt='moment', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:q', wrt='V', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:q', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:q', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:q', wrt='q', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:q', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='dXdt:q', wrt='rw_slope', rows=ar, cols=zz)

        self.declare_partials(of='rf_mainwheel', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='moment', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='V', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='q', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='rf_mainwheel', wrt='rw_slope', rows=ar, cols=zz)

    def compute(self, inputs, outputs, **kwargs):
        thrust = inputs['thrust']
        lift = inputs['lift']
        drag = inputs['drag']
        moment = inputs['moment']
        V = inputs['V']
        mass = inputs['mass']
        alpha = inputs['alpha']
        q = inputs['q']
        grav, = inputs['grav']
        rw_slope, = inputs['rw_slope']

        for out, res in self._compute(thrust, lift, drag, moment, V, mass, alpha, q, grav, rw_slope).items():
            outputs[out] = res

    def compute_partials(self, inputs, partials, *kwargs):
        thrust = inputs['thrust']
        lift = inputs['lift']
        drag = inputs['drag']
        moment = inputs['moment']
        V = inputs['V']
        mass = inputs['mass']
        alpha = inputs['alpha']
        q = inputs['q']
        grav, = inputs['grav']
        rw_slope, = inputs['rw_slope']

        wrt = 'thrust', 'lift', 'drag', 'moment', 'V', 'mass', 'alpha', 'q', 'grav', 'rw_slope'
        args = thrust, lift, drag, moment, V, mass, alpha, q, grav, rw_slope
        for of, grad_fun in self.grad_funcs.items():
            for var, res in zip(wrt, grad_fun(*args)):
                partials[of, var] = np.asarray(res, float)
