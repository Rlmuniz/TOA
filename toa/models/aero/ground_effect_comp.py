import jax
import numpy as np
import openmdao.api as om

from toa.airplanes import AirplaneData


class GroundEffectComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData,
                             desc='Class containing all airplane data')
        self.init_gradients()

    def init_gradients(self):
        gndeff_grad = jax.grad(self._compute)
        self.gndeff_grad = jax.vmap(gndeff_grad)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='h_cg', val=np.zeros(nn),
                       desc='Airplane altitude from runway level', units='m')

        self.add_output(name="gndeff", val=np.ones(nn), desc='Ground effect',
                        units=None)

    def _compute(self, h_cg):
        airplane = self.options['airplane_data']
        ratio = h_cg / airplane.wingspan
        return 33 * ratio ** 1.5 / (1 + 33 * ratio ** 1.5)

    def compute(self, inputs, outputs,**kwargs):
        h_cg = inputs['h_cg']

        outputs['gndeff'] = self._compute(h_cg)

    def compute_partials(self, inputs, partials, **kwargs):
        h_cg = inputs['h_cg']

        partials['gndeff', 'h_cg'] = np.asarray(self.gndeff_grad(h_cg), float)

