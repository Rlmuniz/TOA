import numpy as np
import openmdao.api as om

from toa.data.airplanes.airplanes import Airplanes


class AeroForcesComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=Airplanes,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='CL', shape=(nn,), desc='Lift coefficient', units=None)
        self.add_input(name='CD', shape=(nn,), desc='Drag coefficient', units=None)
        self.add_input(name='Cm', shape=(nn,), desc='Moment coefficient', units=None)
        self.add_input(name='qbar', shape=(nn,), desc='Dynamic pressure', units='Pa')

        # Outputs
        self.add_output(name='L', shape=(nn,), desc='Lift coefficient', units='N')
        self.add_output(name='D', shape=(nn,), desc='Drag coefficient', units='N')
        self.add_output(name='M', shape=(nn,), desc='Moment coefficient', units='N*m')

    def setup_partials(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.declare_partials(of='L', wrt='CL', rows=ar, cols=ar)
        self.declare_partials(of='L', wrt='qbar', rows=ar, cols=ar)

        self.declare_partials(of='D', wrt='CD', rows=ar, cols=ar)
        self.declare_partials(of='D', wrt='qbar', rows=ar, cols=ar)

        self.declare_partials(of='M', wrt='Cm', rows=ar, cols=ar)
        self.declare_partials(of='M', wrt='qbar', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']
        qS = inputs['qbar'] * airplane.wing.area

        outputs['L'] = qS * inputs['CL']
        outputs['D'] = qS * inputs['CD']
        outputs['M'] = qS * airplane.wing.mac * inputs['Cm']

    def compute_partials(self, inputs, partials, **kwargs):
        airplane = self.options['airplane_data']
        S = airplane.wing.area
        qS = inputs['qbar'] * S

        partials['L', 'CL'] = qS
        partials['L', 'qbar'] = S * inputs['CL']

        partials['D', 'CD'] = qS
        partials['D', 'qbar'] = S * inputs['CD']

        partials['M', 'Cm'] = qS * airplane.wing.mac
        partials['M', 'qbar'] = airplane.wing.mac * S * inputs['Cm']
