import numpy as np
import openmdao.api as om

from toa.airplanes import AirplaneData


class www(om.ExplicitComponent):
    """Computes the cg and tdp position. """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input(name='x', shape=(nn,), desc='X distance from brake release', units='m')
        self.add_input(name='h', shape=(nn,), desc='H distance from runway level', units='m')

        # Outputs
        self.add_output(name='x_cg', val=np.zeros(nn), desc='X distance from brake release', units='m')
        self.add_output(name='h_cg', val=np.zeros(nn), desc='H CG distance from runway level', units='m')
        self.add_output(name='x_mlg', val=np.zeros(nn), desc='X main landing gear distance from brake release', units='m')
        self.add_output(name='h_mlg', val=np.zeros(nn), desc='H main landing gear distance from runway level', units='m')

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='q', wrt='rho', rows=ar, cols=ar)
        self.declare_partials(of='q', wrt='tas', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options["airplane_data"]
        x = inputs['x']
        h = inputs['h']

        outputs['x_cg'] = x + airplane.xm
        outputs['h_cg'] = h + airplane.h_cg
        outputs['x_mlg'] = x
        outputs['h_mlg'] = h

    def compute_partials(self, inputs, partials, **kwargs):
        partials['q', 'rho'] = 0.5 * inputs['tas'] ** 2
        partials['q', 'tas'] = inputs['rho'] * inputs['tas']