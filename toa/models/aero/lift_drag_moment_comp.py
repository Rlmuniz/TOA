import numpy as np
import openmdao.api as om

from toa.data import Airplane
from toa.data import get_airplane_data


class LiftDragMomentComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zz = np.zeros(nn)
        ones = np.ones(nn)

        self.add_input(name='CL', val=ones, desc='Lift coefficient', units=None)
        self.add_input(name='CD', val=ones, desc='Drag coefficient', units=None)
        self.add_input(name='Cm', val=ones, desc='Moment coefficient', units=None)
        self.add_input(name='qbar', val=ones, desc='Dynamic pressure', units='Pa')

        # Outputs
        self.add_output(name='L', val=zz, desc='Lift coefficient', units='N')
        self.add_output(name='D', val=zz, desc='Drag coefficient', units='N')
        self.add_output(name='M', val=zz, desc='Moment coefficient', units='N*m')

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


if __name__ == '__main__':
    prob = om.Problem()
    airplane = get_airplane_data('b734')
    num_nodes = 1
    prob.model.add_subsystem('comp', LiftDragMomentComp(num_nodes=1, airplane=airplane))

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    prob.check_partials(compact_print=True, show_only_incorrect=True)