import numpy as np
import openmdao.api as om


class MachComp(om.ExplicitComponent):
    """Computes mach."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='tas', shape=(nn,), desc='True airspeed', units='m/s')
        self.add_input(name='sos', shape=(1,), desc='Atmospheric speed of sound', units='m/s')

        self.add_output(name='mach', shape=(nn,), desc='Mach number', units=None)

    def setup_partials(self):
        self.declare_partials(of='mach', wrt='tas')
        self.declare_partials(of='mach', wrt='sos')

    def compute(self, inputs, outputs, **kwargs):
        outputs['mach'] = inputs['tas'] / inputs['sos']

    def compute_partials(self, inputs, partials, **kwargs):
        partials['mach', 'sos'] = -inputs['tas'] / inputs['sos'] ** 2
        partials['mach', 'tas'] = 1 / inputs['sos']
