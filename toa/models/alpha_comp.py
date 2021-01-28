import numpy as np
import openmdao.api as om

class AlphaComp(om.ExplicitComponent):
    """Computes thrust ratio considering effects of altitude and speed."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        # Inputs
        self.add_input(name='theta', shape=(nn,), desc='Pitch angle', units='rad')
        self.add_input(name='gam', shape=(nn,), desc='Flight path angle', units='rad')

        # Outputs
        self.add_output(name='alpha', val=np.zeros(nn), desc='Angle of Attack',
                        units='rad')

        self.declare_partials(of='alpha', wrt='theta', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='alpha', wrt='gam', rows=ar, cols=ar, val=-1.0)

    def compute(self, inputs, outputs, **kwargs):
        theta = inputs['theta']
        gam = inputs['gam']

        outputs['alpha'] = theta - gam
