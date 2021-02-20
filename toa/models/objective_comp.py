import numpy as np
import openmdao.api as om

class ObjectiveComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        ones = np.ones(nn)

        self.add_input(name='mass', val=ones, desc='Airplane mass', units='kg')
        self.add_input(name='x', val=ones, desc='X cg distance from brake release', units='m')

        self.add_output(name='obj', val=ones, desc='Objective function', units=None)

        self.declare_partials(of='obj', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='obj', wrt='x', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        mass = inputs['mass']
        x = inputs['x']

        outputs['obj'] = mass / x

    def compute_partials(self, inputs, partials, **kwargs):
        mass = inputs['mass']
        x = inputs['x']

        partials['obj', 'mass'] = 1 / x
        partials['obj', 'x'] = -mass / x ** 2
