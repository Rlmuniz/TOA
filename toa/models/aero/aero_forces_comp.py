import numpy as np
import openmdao.api as om
from toa.airplanes import AirplaneData


class AeroForcesComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='CL', shape=(nn,), desc='Lift coefficient', units=None)
        self.add_input(name='CD', shape=(nn,), desc='Drag coefficient', units=None)
        self.add_input(name='Cm', shape=(nn,), desc='Moment coefficient', units=None)
        self.add_input(name='qbar', val=np.zeros(nn), desc='Dynamic pressure', units='Pa')

        # Outputs
        self.add_output(name='L', shape=(nn,), desc='Lift coefficient', units='N')
        self.add_output(name='D', shape=(nn,), desc='Drag coefficient', units='N')
        self.add_output(name='M', shape=(nn,), desc='Moment coefficient', units='N')

        # partials
        ar = np.arange(nn)
        self.declare_partials(of='L', wrt='CL', rows=ar, cols=ar)
        self.declare_partials(of='L', wrt='qbar', rows=ar, cols=ar)
        self.declare_partials(of='D', wrt='CD', rows=ar, cols=ar)
        self.declare_partials(of='D', wrt='qbar', rows=ar, cols=ar)
        self.declare_partials(of='M', wrt='Cm', rows=ar, cols=ar)
        self.declare_partials(of='M', wrt='qbar', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane']
        qS = inputs['qbar'] * airplane.S

        outputs['L'] = qS * inputs['CL']
        outputs['D'] = qS * inputs['CD']
        outputs['M'] = qS * airplane.cbar * inputs['Cm']

    def compute_partials(self, inputs, partials, **kwargs):
        airplane = self.options['airplane']

        qS = inputs['qbar'] * airplane.S

        partials['L', 'CL'] = qS
        partials['L', 'q'] = airplane.S * inputs['CL']

        partials['D', 'CD'] = qS
        partials['D', 'q'] = airplane.S * inputs['CD']

        partials['M', 'Cm'] = qS * airplane.cbar
        partials['M', 'q'] = airplane.cbar * airplane.S * inputs['Cm']