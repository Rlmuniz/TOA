import numpy as np
import openmdao.api as om

from toa.data import Airplane


class LiftDragMomentComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input(name='CL', shape=(nn,), desc='Lift coefficient', units=None)
        self.add_input(name='CD', shape=(nn,), desc='Drag coefficient', units=None)
        self.add_input(name='Cm', shape=(nn,), desc='Moment coefficient', units=None)
        self.add_input(name='qbar', shape=(nn,), desc='Dynamic pressure', units='Pa')

        # Outputs
        self.add_output(name='L', shape=(nn,), desc='Lift coefficient', units='N')
        self.add_output(name='D', shape=(nn,), desc='Drag coefficient', units='N')
        self.add_output(name='M', shape=(nn,), desc='Moment coefficient', units='N*m')

        self.declare_partials(of='L', wrt='CL', rows=ar, cols=ar)
        self.declare_partials(of='L', wrt='qbar', rows=ar, cols=ar)

        self.declare_partials(of='D', wrt='CD', rows=ar, cols=ar)
        self.declare_partials(of='D', wrt='qbar', rows=ar, cols=ar)

        self.declare_partials(of='M', wrt='Cm', rows=ar, cols=ar)
        self.declare_partials(of='M', wrt='qbar', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        ap = self.options['airplane']
        qS = inputs['qbar'] * ap.wing.area

        outputs['L'] = qS * inputs['CL']
        outputs['D'] = qS * inputs['CD']
        outputs['M'] = qS * ap.wing.mac * inputs['Cm']

    def compute_partials(self, inputs, partials, **kwargs):
        ap = self.options['airplane']
        S = ap.wing.area
        qS = inputs['qbar'] * S

        partials['L', 'CL'] = qS
        partials['L', 'qbar'] = S * inputs['CL']

        partials['D', 'CD'] = qS
        partials['D', 'qbar'] = S * inputs['CD']

        partials['M', 'Cm'] = qS * ap.wing.mac
        partials['M', 'qbar'] = ap.wing.mac * S * inputs['Cm']
