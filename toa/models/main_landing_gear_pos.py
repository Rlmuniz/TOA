import numpy as np
import openmdao.api as om

from toa.data import Airplane


class MainLandingGearPosComp(om.ExplicitComponent):
    """Computes the main landing gear position."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input(name='x', val=np.zeros(nn), desc='X cg distance from brake release', units='m')
        self.add_input(name='h', val=np.zeros(nn), desc='H cg distance from runway level', units='m')
        self.add_input(name='theta', val=np.zeros(nn), desc='Pitch Angle', units='rad')

        # Outputs
        self.add_output(name='x_mlg', val=np.zeros(nn), desc='X mlg distance from brake release', units='m')
        self.add_output(name='h_mlg', val=np.zeros(nn), desc='H mlg distance from runway level', units='m')

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='x_mlg', wrt=['*'], method='fd')
        self.declare_partials(of='h_mlg', wrt=['*'], method='fd')

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane']
        x = inputs['x']
        h = inputs['h']
        theta = inputs['theta']

        x1 = -airplane.landing_gear.main.x
        h1 = -airplane.landing_gear.main.z

        x2 = x1*np.cos(theta) - h1*np.sin(theta)
        h2 = x1*np.sin(theta) + h1*np.cos(theta)

        outputs['x_mlg'] = x + x2
        outputs['h_mlg'] = h + h2