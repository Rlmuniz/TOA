import numpy as np
import openmdao.api as om


class DynamicPressureComp(om.ExplicitComponent):
    """Compute the dynamic pressure based on the velocity and the atmospheric density. """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input(name='rho', shape=(nn,), desc='Atmospheric density', units='kg/m**3')
        self.add_input(name='tas', shape=(nn,), desc='True airspeed', units='m/s')

        # Outputs
        self.add_output(name='qbar', val=np.zeros(nn), desc='Dynamic pressure', units='Pa')

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='qbar', wrt='rho', rows=ar, cols=ar)
        self.declare_partials(of='qbar', wrt='tas', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        outputs['qbar'] = 0.5 * inputs['rho'] * inputs['tas'] ** 2

    def compute_partials(self, inputs, partials, **kwargs):
        partials['qbar', 'rho'] = 0.5 * inputs['tas'] ** 2
        partials['qbar', 'tas'] = inputs['rho'] * inputs['tas']