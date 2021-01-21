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

        self.add_input(name='alpha', shape=(1,), desc='Angle of attack', units='rad')
        self.add_input(name='de', shape=(1,), desc='Elevator angle', units='rad')
        self.add_input(name='dih', shape=(1,), desc='Horizontal stabilizer angle', units='rad')

        self.add_output(name='CL', val=np.zeros(nn), desc='Lift coefficient',
                        units=None)

        self.declare_partials(of='CL', wrt='alpha', rows=ar, cols=zz,
                              val=ap.coeffs.CLa)
        self.declare_partials(of='CL', wrt='de', rows=ar, cols=zz,
                              val=ap.coeffs.CLde)
        self.declare_partials(of='CL', wrt='dih', rows=ar, cols=zz,
                              val=ap.coeffs.CLih)

    def compute(self, inputs, outputs, **kwargs):
        ap = self.options['airplane']
        alpha = inputs['alpha']
        de = inputs['de']
        dih = inputs['dih']

        outputs['CL'] = ap.coeffs.CL0 + ap.coeffs.CLa * alpha + ap.coeffs.CLde * de + ap.coeffs.CLih * dih


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
        self.add_input(name='dih', shape=(1,), desc='Horizontal stabilizer angle',
                       units='rad')

        self.add_output(name='CL', val=np.zeros(nn), desc='Lift coefficient', units=None)

        self.declare_partials(of='CL', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='de', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='tas', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='q', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='dih', rows=ar, cols=np.zeros(nn))

    def compute(self, inputs, outputs, **kwargs):
        ap = self.options['airplane']
        alpha = inputs['alpha']
        de = inputs['de']
        q = inputs['q']
        tas = inputs['tas']
        dih = inputs['dih']

        qhat = q * ap.wing.mac / (2 * tas)

        outputs['CL'] = (
                ap.coeffs.CL0
                + ap.coeffs.CLa * alpha
                + ap.coeffs.CLde * de
                + ap.coeffs.CLq * qhat
                + ap.coeffs.CLih * dih
        )

    def compute_partials(self, inputs, partials, **kwargs):
        ap = self.options['airplane']
        q = inputs['q']
        tas = inputs['tas']

        partials['CL', 'alpha'] = ap.coeffs.CLa
        partials['CL', 'de'] = ap.coeffs.CLde
        partials['CL', 'q'] = ap.coeffs.CLq * ap.wing.mac / (2 * tas)
        partials['CL', 'tas'] = - ap.coeffs.CLq * q * ap.wing.mac / (2 * tas ** 2)
        partials['CL', 'dih'] = ap.coeffs.CLih
