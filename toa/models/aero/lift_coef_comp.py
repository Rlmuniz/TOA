import numpy as np
import openmdao.api as om

from toa.data import Airplane


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

        self.add_output(name='CL', val=np.zeros(nn), desc='Lift coefficient',
                        units=None)

        self.declare_partials(of='CL', wrt='alpha', rows=ar, cols=ar,
                              val=ap.coeffs.CLa)
        self.declare_partials(of='CL', wrt='de', rows=ar, cols=ar,
                              val=ap.coeffs.CLde)

    def compute(self, inputs, outputs, **kwargs):
        ap = self.options['airplane']
        alpha = inputs['alpha']
        de = inputs['de']

        outputs['CL'] = ap.coeffs.CL0 + ap.coeffs.CLa * alpha + ap.coeffs.CLde * de


class LiftCoeffComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane', types=Airplane,
                             desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input(name='alpha', shape=(nn,), desc='Angle of attack', units='rad')
        self.add_input(name='de', shape=(nn,), desc='Elevator angle', units='rad')
        self.add_input(name='tas', shape=(nn,), desc='True Airspeed', units='m/s')
        self.add_input(name='q', shape=(nn,), desc='Pitch Rate', units='rad/s')

        self.add_output(name='CL', val=np.zeros(nn), desc='Lift coefficient', units=None)

        self.declare_partials(of='CL', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='de', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='tas', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='q', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        ap = self.options['airplane']
        alpha = inputs['alpha']
        de = inputs['de']
        q = inputs['q']
        tas = inputs['tas']

        qhat = q * ap.wing.mac / (2 * tas)

        outputs['CL'] = (
                ap.coeffs.CL0
                + ap.coeffs.CLa * alpha
                + ap.coeffs.CLde * de
                + ap.coeffs.CLq * qhat
        )

    def compute_partials(self, inputs, partials, **kwargs):
        ap = self.options['airplane']
        q = inputs['q']
        tas = inputs['tas']

        partials['CL', 'alpha'] = ap.coeffs.CLa
        partials['CL', 'de'] = ap.coeffs.CLde
        partials['CL', 'q'] = ap.coeffs.CLq * ap.wing.mac / (2 * tas)
        partials['CL', 'tas'] = - ap.coeffs.CLq * q * ap.wing.mac / (2 * tas ** 2)
