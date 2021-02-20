import openmdao.api as om
import numpy as np


class LiftCoeffGroundCorrectionComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zz = np.zeros(nn)

        self.add_input(name='CL', val=zz, desc='Lift coefficient', units=None)
        self.add_input(name='CLa', val=1.0, desc='Lift x alfa curve slope', units='1/rad')
        self.add_input(name="CLag", val=zz, desc='CLa variation due to ground effect', units='1/rad')
        self.add_input(name='dalpha_zero', val=zz, desc='Alpha zero CL variation due to ground effect',
                       units='rad')

        self.add_output(name='CLg', val=np.zeros(nn), desc='Lift coefficient', units=None)

        self.declare_partials(of='CLg', wrt='CL', rows=ar, cols=ar)
        self.declare_partials(of='CLg', wrt='CLa', rows=ar, cols=zz)
        self.declare_partials(of='CLg', wrt='CLag', rows=ar, cols=ar)
        self.declare_partials(of='CLg', wrt='dalpha_zero', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        CL = inputs['CL']
        CLa = inputs['CLa']
        CLag = inputs['CLag']
        dalpha_zero = inputs['dalpha_zero']

        outputs['CLg'] = CL * CLag / CLa - CLag * dalpha_zero

    def compute_partials(self, inputs, partials, **kwargs):
        CL = inputs['CL']
        CLa = inputs['CLa']
        CLag = inputs['CLag']
        dalpha_zero = inputs['dalpha_zero']

        partials['CLg', 'CL'] = CLag / CLa
        partials['CLg', 'CLa'] = -CL * CLag / CLa ** 2
        partials['CLg', 'CLag'] = CL / CLa - dalpha_zero
        partials['CLg', 'dalpha_zero'] = -CLag


if __name__ == '__main__':
    prob = om.Problem()
    num_nodes = 1
    prob.model.add_subsystem('comp', LiftCoeffGroundCorrectionComp(num_nodes=1))

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    prob.check_partials(compact_print=True, show_only_incorrect=True)