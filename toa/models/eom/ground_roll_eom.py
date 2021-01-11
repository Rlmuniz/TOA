import jax
import numpy as np
import jax.numpy as jnp
import openmdao.api as om

from toa.data.airplanes.airplanes import Airplanes


class GroundRollEOM(om.ExplicitComponent):
    """Computes the ground run (1 DoF) with all wheels on the runway to the point of nose wheels lift-off."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=Airplanes, desc='Class containing all airplane data')
        self._init_gradients()

    def _init_gradients(self):
        """Generates function gradients for component."""
        argnums = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        in_axes = (0, 0, 0, 0, 0, 0, None, None, None)

        dXdt_v_grad = jax.vmap(jax.grad(self._compute_dXdt_v, argnums), in_axes=in_axes)
        rf_nosewheel_grad = jax.vmap(jax.grad(self._compute_rf_nosewheel, argnums), in_axes=in_axes)
        rf_mainwheel_grad = jax.vmap(jax.grad(self._compute_rf_mainwheel, argnums), in_axes=in_axes)

        self.grad_funcs = {
            'dXdt:v': dXdt_v_grad,
            'rf_nosewheel': rf_nosewheel_grad,
            'rf_mainwheel': rf_mainwheel_grad
        }

    def _compute_dXdt_v(self, thrust, lift, drag, moment, V, mass,
                        grav, rw_slope, alpha):
        return self._compute(thrust, lift, drag, moment, V, mass, grav, rw_slope, alpha)['dXdt:v']

    def _compute_rf_nosewheel(self, thrust, lift, drag, moment, V, mass, grav, rw_slope, alpha):
        return self._compute(thrust, lift, drag, moment, V, mass, grav, rw_slope, alpha)['rf_nosewheel']

    def _compute_rf_mainwheel(self, thrust, lift, drag, moment, V, mass, grav, rw_slope, alpha):
        return self._compute(thrust, lift, drag, moment, V, mass, grav, rw_slope, alpha)['rf_mainwheel']

    def _compute(self, thrust, lift, drag, moment, V, mass, grav, rw_slope, alpha):
        airplane = self.options['airplane_data']
        mu = 0.002

        weight = mass * grav
        rf_nosewheel = (moment + thrust * airplane.engine.zpos + airplane.landing_gear.x_mg * (
                    weight * jnp.cos(rw_slope) - lift)) / (
                               airplane.landing_gear.x_mg - airplane.landing_gear.x_ng)
        rf_mainwheel = (moment + thrust * airplane.engine.zpos + airplane.landing_gear.x_ng * (
                    weight * jnp.cos(rw_slope) - lift)) / (
                               airplane.landing_gear.x_ng - airplane.landing_gear.x_mg)

        f_rr = mu * (rf_nosewheel + rf_mainwheel)

        return {
            'dXdt:v': (thrust * jnp.cos(alpha) - drag - f_rr - weight * jnp.sin(
                    rw_slope)) / mass,
            'dXdt:x': V,
            'rf_nosewheel': rf_nosewheel,
            'rf_mainwheel': rf_mainwheel
        }

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input(name='thrust', val=np.zeros(nn), desc='Engine total thrust', units='N')
        self.add_input(name='lift', val=np.zeros(nn), desc='Lift', units='N')
        self.add_input(name='drag', val=np.zeros(nn), desc='Drag force', units='N')
        self.add_input(name='moment', val=np.zeros(nn), desc='Aerodynamic moment', units='N*m')
        self.add_input(name='V', val=np.zeros(nn), desc='Body x axis velocity', units='m/s')
        self.add_input(name='mass', val=np.zeros(nn), desc='Airplane mass', units='kg')
        self.add_input(name='rw_slope', val=0.0, desc='Runway slope', units='rad')
        self.add_input(name='grav', val=0.0, desc='Gravity acceleration', units='m/s**2')
        self.add_input(name='alpha', val=0.0, desc='Angle of attack', units='rad')

        # Outputs
        self.add_output(name='dXdt:v', val=np.zeros(nn), desc="Body x axis acceleration", units='m/s**2')
        self.add_output(name='dXdt:x', val=np.zeros(nn), desc="Derivative of position", units='m/s')
        self.add_output(name='rf_nosewheel', val=np.zeros(nn), desc="Nose wheel reaction force", units='N')
        self.add_output(name='rf_mainwheel', val=np.zeros(nn), desc="Main wheel reaction force", units='N')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zz = np.zeros(nn)
        ones = np.ones(nn)

        self.declare_partials(of='dXdt:v', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='moment', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='V', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:v', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='dXdt:v', wrt='rw_slope', rows=ar, cols=zz)
        self.declare_partials(of='dXdt:v', wrt='alpha', rows=ar, cols=zz)

        self.declare_partials(of='dXdt:x', wrt='V', rows=ar, cols=ar, val=ones)

        self.declare_partials(of='rf_nosewheel', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='rf_nosewheel', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='rf_nosewheel', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='rf_nosewheel', wrt='moment', rows=ar, cols=ar)
        self.declare_partials(of='rf_nosewheel', wrt='V', rows=ar, cols=ar)
        self.declare_partials(of='rf_nosewheel', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='rf_nosewheel', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='rf_nosewheel', wrt='rw_slope', rows=ar, cols=zz)
        self.declare_partials(of='rf_nosewheel', wrt='alpha', rows=ar, cols=zz)

        self.declare_partials(of='rf_mainwheel', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='moment', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='V', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='rf_mainwheel', wrt='grav', rows=ar, cols=zz)
        self.declare_partials(of='rf_mainwheel', wrt='rw_slope', rows=ar, cols=zz)
        self.declare_partials(of='rf_mainwheel', wrt='alpha', rows=ar, cols=zz)

    def compute(self, inputs, outputs, **kwargs):
        thrust = inputs['thrust']
        lift = inputs['lift']
        drag = inputs['drag']
        moment = inputs['moment']
        V = inputs['V']
        mass = inputs['mass']
        grav, = inputs['grav']
        rw_slope, = inputs['rw_slope']
        alpha, = inputs['alpha']

        for out, res in self._compute(thrust, lift, drag, moment, V, mass, grav, rw_slope, alpha).items():
            outputs[out] = res

    def compute_partials(self, inputs, partials, **kwargs):
        thrust = inputs['thrust']
        lift = inputs['lift']
        drag = inputs['drag']
        moment = inputs['moment']
        V = inputs['V']
        mass = inputs['mass']
        grav, = inputs['grav']
        rw_slope, = inputs['rw_slope']
        alpha, = inputs['alpha']

        wrt = 'thrust', 'lift', 'drag', 'moment', 'V', 'mass', 'grav', 'rw_slope', 'alpha'
        args = thrust, lift, drag, moment, V, mass, grav, rw_slope, alpha
        for of, grad_fun in self.grad_funcs.items():
            for var, res in zip(wrt, grad_fun(*args)):
                partials[of, var] = np.asarray(res, float)
