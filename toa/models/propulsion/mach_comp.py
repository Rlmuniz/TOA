import numpy as np
import openmdao.api as om


class MachComp(om.ExplicitComponent):
    """Computes mach."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zz = np.zeros(nn)
        ones = np.ones(nn)

        self.add_input(name='tas', val=ones, desc='True airspeed', units='m/s')
        self.add_input(name='sos', val=ones, desc='Atmospheric speed of sound',
                       units='m/s')

        self.add_output(name='mach', val=ones, desc='Mach number', units=None)

        self.declare_partials(of='mach', wrt='tas', rows=ar, cols=ar)
        self.declare_partials(of='mach', wrt='sos', rows=ar, cols=zz)

    def compute(self, inputs, outputs, **kwargs):
        outputs['mach'] = inputs['tas'] / inputs['sos']

    def compute_partials(self, inputs, partials, **kwargs):
        partials['mach', 'sos'] = - inputs['tas'] / inputs['sos'] ** 2
        partials['mach', 'tas'] = 1.0 / inputs['sos']


if __name__ == '__main__':
    prob = om.Problem()
    num_nodes = 1
    prob.model.add_subsystem('comp', MachComp(num_nodes=1))

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    prob.check_partials(compact_print=True, show_only_incorrect=True)
