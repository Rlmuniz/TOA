import jax
import numpy as np
import openmdao.api as om

from toa.data.airplanes.airplanes import Airplanes


class FuelFlowComp(om.ExplicitComponent):
    """Computes fuel flow."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('condition', default='AEO', desc='Takeoff condition (AEO/OEI)')
        self.options.declare('airplane_data', types=Airplanes, desc='Class containing all airplane data')
        self._init_gradients()

    def _init_gradients(self):
        """Generates function gradients for component."""
        argnums = (0, 1, 2)
        in_axes = (0, 0, None)

        self.grad_funcs = jax.vmap(jax.grad(self._compute, argnums), in_axes=in_axes)

    def _compute(self, thrust_ratio, thrust, elevation):
        airplane = self.options['airplane_data']

        num_motors = airplane.engine.num_motors if self.options[
                                                       'condition'] == 'AEO' else airplane.engine.num_motors - 1

        ength = thrust / num_motors

        mass_fuel_eng = airplane.engine.cff3 * thrust_ratio ** 3 + airplane.engine.cff2 * thrust_ratio ** 2 + airplane.engine.cff1 * thrust_ratio + 6.7e-7 * (
                ength / 1000) * elevation

        return mass_fuel_eng * num_motors

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='thrust_ratio', shape=(nn,), desc='Thrust ratio', units=None)
        self.add_input(name='thrust', shape=(nn,), desc='Thrust at current elevation and speed', units='N')
        self.add_input(name='elevation', shape=(1,), desc='Runway elevation', units='m')

        self.add_output(name='dXdt:mass_fuel', val=np.zeros(nn),
                        desc='rate of aircraft mass change - negative when fuel is being depleted', units='kg/s')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zz = np.zeros(nn)

        self.declare_partials(of='dXdt:mass_fuel', wrt='thrust_ratio', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_fuel', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:mass_fuel', wrt='elevation', rows=ar, cols=zz)

    def compute(self, inputs, outputs, **kwargs):
        thrust_ratio = inputs['thrust_ratio']
        thrust = inputs['thrust']
        elevation, = inputs['elevation']

        outputs['dXdt:mass_fuel'] = self._compute(thrust_ratio, thrust, elevation)

    def compute_partials(self, inputs, partials, **kwargs):
        thrust_ratio = inputs['thrust_ratio']
        thrust = inputs['thrust']
        elevation, = inputs['elevation']

        wrt = 'thrust_ratio', 'thrust', 'elevation'
        args = thrust_ratio, thrust, elevation
        for _wrt, res in zip(wrt, self.grad_funcs(*args)):
            partials['dXdt:mass_fuel', _wrt] = np.asarray(res, float)
