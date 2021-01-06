import numpy as np
import openmdao.api as om

from toa.airplanes import AirplaneData


class GroundEffectComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='h', val=np.zeros(nn), desc='Airplane altitude from runway level', units='m')

        self.add_output(name="g_effect", val=np.ones(nn), desc='Ground effect', units=None)


    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']
        h = inputs['h']