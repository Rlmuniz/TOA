import numpy as np
import openmdao.api as om


class TrueAirspeedCompGroundRoll(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='Vw', val=0.0, desc='Wind speed along the runway, defined as positive for a headwind', units='m/s')
        self.add_input(name='V', shape=(nn,), desc='Body x axis velocity', units='m/s')

        self.add_output(name='tas', val=np.zeros(nn), desc="True airspeed", units='m/s')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.declare_partials(of='tas', wrt='V', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='tas', wrt='Vw', rows=ar, cols=np.zeros(nn), val=1.0)

    def compute(self, inputs, outputs, **kwargs):
        outputs['tas'] = inputs['V'] + inputs['Vw']
