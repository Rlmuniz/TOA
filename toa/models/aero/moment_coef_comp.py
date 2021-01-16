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
        self.add_input(name='de', shape=(nn,), desc='Elevator angle', units='rad')

        self.add_output(name='Cm', val=zz, desc='Moment coefficient', units=None)

        self.declare_partials(of='Cm', wrt='alpha', rows=ar, cols=zz, val=ap.coeffs.Cma)
        self.declare_partials(of='Cm', wrt='de', rows=ar, cols=ar, val=ap.coeffs.Cmde)

    def compute(self, inputs, outputs, **kwargs):
        ap = self.options['airplane']
        alpha = inputs['alpha']
        de = inputs['de']

        outputs['Cm'] = ap.coeffs.Cm0 + ap.coeffs.Cma * alpha + ap.coeffs.Cmde * de


class MomentCoeffComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        zeros = np.zeros(nn)
        ar = np.arange(nn)

        self.add_input(name='alpha', val=zeros, desc='Angle of attack', units='rad')
        self.add_input(name='de', val=zeros, desc='Elevator angle', units='rad')
        self.add_input(name='tas', val=zeros, desc='True Airspeed', units='m/s')
        self.add_input(name='q', val=zeros, desc='Pitch Rate', units='rad/s')

        self.add_output(name='Cm', val=zeros, desc='Moment coefficient', units=None)

        self.declare_partials(of='Cm', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='Cm', wrt='de', rows=ar, cols=ar)
        self.declare_partials(of='Cm', wrt='tas', rows=ar, cols=ar)
        self.declare_partials(of='Cm', wrt='q', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        ap = self.options['airplane']
        alpha = inputs['alpha']
        de = inputs['de']
        q = inputs['q']
        tas = inputs['tas']

        qhat = q * ap.wing.mac / (2 * tas)

        outputs['Cm'] = (
                ap.coeffs.Cm0
                + ap.coeffs.Cma * alpha
                + ap.coeffs.Cmde * de
                + ap.coeffs.Cmq * qhat
        )

    def compute_partials(self, inputs, partials, **kwargs):
        ap = self.options['airplane']
        q = inputs['q']
        tas = inputs['tas']

        partials['Cm', 'alpha'] = ap.coeffs.Cma
        partials['Cm', 'de'] = ap.coeffs.Cmde
        partials['Cm', 'q'] = ap.coeffs.Cmq * ap.wing.mac / (2 * tas)
        partials['Cm', 'tas'] = - ap.coeffs.Cmq * q * ap.wing.mac / (2 * tas ** 2)
