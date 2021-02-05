import numpy as np
import openmdao.api as om

from toa.data import Airplane
from toa.data import get_airplane_data


class LiftCoeffAllWheelsOnGroundComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        ap = self.options['airplane']
        ar = np.arange(nn)
        zz = np.zeros(nn)

        self.add_input(name='alpha', shape=(nn,), desc='Angle of attack', units='rad')
        self.add_input(name='de', shape=(nn,), desc='Elevator angle', units='rad')
        self.add_input(name='dih', shape=(1,), desc='Horizontal stabilizer angle',
                       units='rad')
        self.add_input(name='CL0', val=ap.coeffs.CL0,
                       desc='Lift coefficient for alpha zero', units=None)
        self.add_input(name='CLa', val=ap.coeffs.CLa, desc='Lift x alfa curve slope',
                       units='1/rad')

        self.add_output(name='CL', val=np.zeros(nn), desc='Lift coefficient',
                        units=None)

        self.declare_partials(of='CL', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='de', rows=ar, cols=ar, val=ap.coeffs.CLde)
        self.declare_partials(of='CL', wrt='dih', rows=ar, cols=zz, val=ap.coeffs.CLih)
        self.declare_partials(of='CL', wrt='CLa', rows=ar, cols=zz)
        self.declare_partials(of='CL', wrt='CL0', rows=ar, cols=zz, val=1.0)

    def compute(self, inputs, outputs, **kwargs):
        ap = self.options['airplane']
        alpha = inputs['alpha']
        de = inputs['de']
        dih = inputs['dih']
        CLa = inputs['CLa']
        CL0 = inputs['CL0']

        outputs['CL'] = CL0 + CLa * alpha + ap.coeffs.CLde * de + ap.coeffs.CLih * dih

    def compute_partials(self, inputs, partials, **kwargs):
        alpha = inputs['alpha']
        CLa = inputs['CLa']

        partials['CL', 'alpha'] = CLa
        partials['CL', 'CLa'] = alpha


class LiftCoeffComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        ap = self.options['airplane']
        ar = np.arange(nn)
        zz = np.zeros(nn)

        self.add_input(name='alpha', shape=(nn,), desc='Angle of attack', units='rad')
        self.add_input(name='de', shape=(nn,), desc='Elevator angle', units='rad')
        self.add_input(name='tas', shape=(nn,), desc='True Airspeed', units='m/s')
        self.add_input(name='q', shape=(nn,), desc='Pitch Rate', units='rad/s')
        self.add_input(name='dih', shape=(1,), desc='Horizontal stabilizer angle',
                       units='rad')
        self.add_input(name='CL0', val=ap.coeffs.CL0,
                       desc='Lift coefficient for alpha zero', units=None)
        self.add_input(name='CLa', val=ap.coeffs.CLa, desc='Lift x alfa curve slope',
                       units='1/rad')

        self.add_output(name='CL', val=np.zeros(nn), desc='Lift coefficient',
                        units=None)

        self.declare_partials(of='CL', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='de', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='tas', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='q', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='dih', rows=ar, cols=zz)
        self.declare_partials(of='CL', wrt='CLa', rows=ar, cols=zz)
        self.declare_partials(of='CL', wrt='CL0', rows=ar, cols=zz, val=1.0)

    def compute(self, inputs, outputs, **kwargs):
        ap = self.options['airplane']
        alpha = inputs['alpha']
        de = inputs['de']
        q = inputs['q']
        tas = inputs['tas']
        dih = inputs['dih']
        CLa = inputs['CLa']
        CL0 = inputs['CL0']

        qhat = q * ap.wing.mac / (2 * tas)

        outputs['CL'] = (
                CL0
                + CLa * alpha
                + ap.coeffs.CLde * de
                + ap.coeffs.CLq * qhat
                + ap.coeffs.CLih * dih
        )

    def compute_partials(self, inputs, partials, **kwargs):
        ap = self.options['airplane']
        q = inputs['q']
        tas = inputs['tas']
        alpha = inputs['alpha']
        CLa = inputs['CLa']

        partials['CL', 'alpha'] = CLa
        partials['CL', 'de'] = ap.coeffs.CLde
        partials['CL', 'q'] = ap.coeffs.CLq * ap.wing.mac / (2 * tas)
        partials['CL', 'tas'] = - ap.coeffs.CLq * q * ap.wing.mac / (2 * tas ** 2)
        partials['CL', 'dih'] = ap.coeffs.CLih
        partials['CL', 'CLa'] = alpha


if __name__ == '__main__':
    prob = om.Problem()
    airplane = get_airplane_data('b734')
    num_nodes = 1
    prob.model.add_subsystem('comp', LiftCoeffComp(num_nodes=1, airplane=airplane))

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    prob.check_partials(compact_print=True, show_only_incorrect=True)

    prob = om.Problem()
    airplane = get_airplane_data('b734')
    num_nodes = 1
    prob.model.add_subsystem('comp', LiftCoeffAllWheelsOnGroundComp(num_nodes=1, airplane=airplane))

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    prob.check_partials(compact_print=True, show_only_incorrect=True)
