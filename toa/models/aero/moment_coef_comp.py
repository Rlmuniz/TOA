import numpy as np
import openmdao.api as om

from toa.data import Airplane


class MomentCoeffAllWheelsOnGroundComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        ap = self.options['airplane']
        ar = np.arange(nn)
        zz = np.zeros(nn)

        self.add_input(name='alpha', shape=(1,), desc='Angle of attack', units='rad')
        self.add_input(name='de', shape=(1,), desc='Elevator angle', units='rad')
        self.add_input(name='dih', shape=(1,), desc='Horizontal stabilizer angle',
                       units='rad')

        self.add_output(name='Cm', val=zz, desc='Moment coefficient', units=None)

        self.declare_partials(of='Cm', wrt='alpha', rows=ar, cols=zz, val=ap.coeffs.Cma)
        self.declare_partials(of='Cm', wrt='de', rows=ar, cols=zz, val=ap.coeffs.Cmde)
        self.declare_partials(of='Cm', wrt='dih', rows=ar, cols=zz, val=ap.coeffs.Cmih)

    def compute(self, inputs, outputs, **kwargs):
        ap = self.options['airplane']
        alpha = inputs['alpha']
        de = inputs['de']
        dih = inputs['dih']

        outputs['Cm'] = ap.coeffs.Cm0 + ap.coeffs.Cma * alpha + ap.coeffs.Cmde * de + ap.coeffs.Cmih * dih


class MomentCoeffComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        ap = self.options['airplane']
        ar = np.arange(nn)

        self.add_input(name='alpha', shape=(nn,), desc='Angle of attack', units='rad')
        self.add_input(name='de', shape=(nn,), desc='Elevator angle', units='rad')
        self.add_input(name='tas', shape=(nn,), desc='True Airspeed', units='m/s')
        self.add_input(name='q', shape=(nn,), desc='Pitch Rate', units='rad/s')
        self.add_input(name='dih', shape=(1,), desc='Horizontal stabilizer angle',
                       units='rad')

        self.add_output(name='Cm', val=np.zeros(nn), desc='Moment coefficient', units=None)

        self.declare_partials(of='Cm', wrt='alpha', rows=ar, cols=ar, val=ap.coeffs.Cma)
        self.declare_partials(of='Cm', wrt='de', rows=ar, cols=ar, val=ap.coeffs.Cmde)
        self.declare_partials(of='Cm', wrt='tas', rows=ar, cols=ar)
        self.declare_partials(of='Cm', wrt='q', rows=ar, cols=ar)
        self.declare_partials(of='Cm', wrt='dih', rows=ar, cols=np.zeros(nn), val=ap.coeffs.Cmih)

    def compute(self, inputs, outputs, **kwargs):
        ap = self.options['airplane']
        alpha = inputs['alpha']
        de = inputs['de']
        q = inputs['q']
        tas = inputs['tas']
        dih = inputs['dih']

        qhat = q * ap.wing.mac / (2 * tas)

        outputs['Cm'] = (
                ap.coeffs.Cm0
                + ap.coeffs.Cma * alpha
                + ap.coeffs.Cmde * de
                + ap.coeffs.Cmq * qhat
                + ap.coeffs.Cmih * dih
        )

    def compute_partials(self, inputs, partials, **kwargs):
        ap = self.options['airplane']
        q = inputs['q']
        tas = inputs['tas']

        partials['Cm', 'q'] = ap.coeffs.Cmq * ap.wing.mac / (2 * tas)
        partials['Cm', 'tas'] = - ap.coeffs.Cmq * q * ap.wing.mac / (2 * tas ** 2)
