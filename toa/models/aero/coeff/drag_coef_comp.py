import jax
import jax.numpy as jnp
import numpy as np
import openmdao.api as om

from toa.data.airplanes.airplanes import Airplanes


class DragCoeffComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=Airplanes, desc='Class containing all airplane data')
        self.options.declare('landing_gear', default=True, desc='Accounts landing gear drag')
        self._init_gradients()

    def _init_gradients(self):
        """Generates function gradients for component."""
        argnums = (0, 1, 2, 3)
        in_axes = (0, 0, None, None)

        self.grad_func = jax.vmap(jax.grad(self._compute, argnums), in_axes=in_axes)

    def _compute(self, mass, CL, flap_angle, grav):
        airplane = self.options['airplane_data']

        delta_cd_flap = airplane.flap.lambda_f * airplane.flap.cf_c ** 1.38 * airplane.flap.sf_s * jnp.sin(flap_angle) ** 2

        if self.options['landing_gear']:
            delta_cd_gear = (mass * grav) / airplane.wing.area * 3.16e-5 * airplane.limits.MTOW ** (-0.215)
        else:
            delta_cd_gear = 0

        CD0_total = airplane.polar.CD0 + delta_cd_gear + delta_cd_flap

        if airplane.engine.mount == 'rear':
            delta_e_flap = 0.0046 * flap_angle
        else:
            delta_e_flap = 0.0026 * flap_angle

        k_total = 1 / (1 / airplane.polar.k + np.pi * airplane.wing.aspect_ratio * delta_e_flap)

        return CD0_total * k_total * CL ** 2

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='flap_angle', shape=(1,), desc='Flap deflection', units='rad')
        self.add_input(name='CL', shape=(nn,), desc='Lift coefficient', units=None)
        self.add_input(name='mass', shape=(nn,), desc='Airplane mass', units='kg')
        self.add_input(name='grav', shape=(1,), desc='Airplane mass', units='m/s**2')

        self.add_output(name='CD', shape=(nn,), desc='Drag coefficient', units=None)

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zz = np.zeros(nn)

        self.declare_partials(of='CD', wrt='flap_angle', rows=ar, cols=zz)
        self.declare_partials(of='CD', wrt='CL', rows=ar, cols=ar)
        self.declare_partials(of='CD', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='CD', wrt='grav', rows=ar, cols=zz)

    def compute(self, inputs, outputs, **kwargs):
        flap_angle, = inputs['flap_angle']
        CL = inputs['CL']
        mass = inputs['mass']
        grav, = inputs['grav']

        outputs['CD'] = self._compute(mass, CL, flap_angle, grav)

    def compute_partials(self, inputs, partials, **kwargs):
        flap_angle, = inputs['flap_angle']
        CL = inputs['CL']
        mass = inputs['mass']
        grav, = inputs['grav']

        wrt = 'mass', 'CL', 'flap_angle', 'grav'
        args = mass, CL, flap_angle, grav
        for _wrt, res in zip(wrt, self.grad_func(*args)):
            partials['CD', _wrt] = np.asarray(res, float)
