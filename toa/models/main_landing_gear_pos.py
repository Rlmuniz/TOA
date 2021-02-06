import numpy as np
import openmdao.api as om

from toa.data import Airplane
from toa.data import get_airplane_data


class MainLandingGearPosComp(om.ExplicitComponent):
    """Computes the main landing gear position."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        ones = np.ones(nn)

        # Inputs
        self.add_input(name='x', val=ones, desc='X cg distance from brake release', units='m')
        self.add_input(name='h', val=ones, desc='H cg distance from runway level', units='m')
        self.add_input(name='theta', val=ones, desc='Pitch Angle', units='rad')

        # Outputs
        self.add_output(name='x_mlg', val=np.zeros(nn), desc='X mlg distance from brake release', units='m')
        self.add_output(name='h_mlg', val=np.zeros(nn), desc='H mlg distance from runway level', units='m')

        # Partials
        ar = np.arange(nn)
        self.declare_partials(of='x_mlg', wrt='x', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='x_mlg', wrt='theta', rows=ar, cols=ar)

        self.declare_partials(of='h_mlg', wrt='h', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='h_mlg', wrt='theta', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane']
        x = inputs['x']
        h = inputs['h']
        theta = inputs['theta']

        x1 = -airplane.landing_gear.main.x
        h1 = -airplane.landing_gear.main.z

        x2 = x1 * np.cos(theta) - h1 * np.sin(theta)
        h2 = x1 * np.sin(theta) + h1 * np.cos(theta)

        outputs['x_mlg'] = x + x2
        outputs['h_mlg'] = h + h2

    def compute_partials(self, inputs, partials, **kwargs):
        airplane = self.options['airplane']
        theta = inputs['theta']

        x1 = -airplane.landing_gear.main.x
        h1 = -airplane.landing_gear.main.z

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        partials['x_mlg', 'theta'] = -h1 * costheta - x1 * sintheta

        partials['h_mlg', 'theta'] = -h1 * sintheta + x1 * costheta


if __name__ == '__main__':
    prob = om.Problem()
    airplane = get_airplane_data('b734')
    num_nodes = 1
    prob.model.add_subsystem('comp', MainLandingGearPosComp(num_nodes=1, airplane=airplane))

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    prob.check_partials(compact_print=True, show_only_incorrect=True)
