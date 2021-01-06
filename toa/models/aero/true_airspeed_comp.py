import numpy as np
import openmdao.api as om


class TrueAirspeedComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='vw', val=0.0, desc='Wind speed along the runway, defined as positive for a headwind', units='m/s')
        self.add_input(name='v', val=np.zeros(nn), desc='Ground speed', units='m/s')

        self.add_output(name='tas', val=np.zeros(nn), desc="True airspeed", units='m/s')

        ar = np.arange(nn)
        self.declare_partials(of='tas', wrt='v', rows=ar, cols=np.zeros(nn), val=1.0)

    def compute(self, inputs, outputs, **kwargs):
        outputs['tas'] = inputs['v'] + inputs['vw']
