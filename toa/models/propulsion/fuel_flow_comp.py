import numpy as np
import openmdao.api as om

from toa.data import Airplane
from toa.data import get_airplane_data


class FuelFlowComp(om.ExplicitComponent):
    """Computes fuel flow."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('condition', default='AEO',
                             desc='Takeoff condition (AEO/OEI)')
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input(name='thrust_ratio', val=np.zeros(nn), desc='Thrust ratio',
                       units=None)
        self.add_input(name='thrust', val=np.zeros(nn),
                       desc='Thrust at current elevation and speed', units='N')
        self.add_input(name='elevation', val=0.0, desc='Runway elevation', units='m')

        self.add_output(name='m_dot', val=np.zeros(nn),
                        desc='rate of aircraft mass change - negative when fuel is being depleted',
                        units='kg/s')

        self.declare_partials(of='m_dot', wrt='thrust_ratio', rows=ar, cols=ar)
        self.declare_partials(of='m_dot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='m_dot', wrt='elevation', rows=ar, cols=np.zeros(nn))

    def compute(self, inputs, outputs, **kwargs):
        thrust_ratio = inputs['thrust_ratio']
        thrust = inputs['thrust']
        elevation = inputs['elevation']

        ap = self.options['airplane']

        if self.options['condition'] == 'AEO':
            num_motors = ap.engine.num_motors
        else:
            num_motors = ap.engine.num_motors - 1

        ength = thrust / num_motors

        mass_fuel_eng = (
                ap.engine.cff3 * thrust_ratio ** 3
                + ap.engine.cff2 * thrust_ratio ** 2
                + ap.engine.cff1 * thrust_ratio + 6.7e-7
                * (ength / 1000) * elevation
        )

        outputs['m_dot'] = -1.0 * mass_fuel_eng * num_motors

    def compute_partials(self, inputs, partials, **kwargs):
        airplane = self.options['airplane']
        elevation = inputs['elevation']
        thrust_ratio = inputs['thrust_ratio']
        thrust = inputs['thrust']

        if self.options['condition'] == 'AEO':
            num_motors = airplane.engine.num_motors
        else:
            num_motors = airplane.engine.num_motors - 1

        partials['m_dot', 'thrust'] = -6.7e-10 * elevation
        partials['m_dot', 'thrust_ratio'] = -num_motors * (
                1.0 * airplane.engine.cff1 + 2.0 * airplane.engine.cff2 * thrust_ratio
                + 3.0 * airplane.engine.cff3 * thrust_ratio ** 2)
        partials['m_dot', 'elevation'] = -6.7e-10 * thrust


if __name__ == '__main__':
    prob = om.Problem()
    airplane = get_airplane_data('b734')
    num_nodes = 1
    prob.model.add_subsystem('comp', FuelFlowComp(num_nodes=1, airplane=airplane))

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    prob.check_partials(compact_print=True, show_only_incorrect=True)
