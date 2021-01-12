import numpy as np
import openmdao.api as om

from toa.data.airplanes.airplanes import Airplanes


class MomentCoeffAllWheelsOnGroundComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=Airplanes, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='alpha', shape=(1,), desc='Angle of attack', units='rad')
        self.add_input(name='de', shape=(nn,), desc='Elevator angle', units='rad')

        self.add_output(name='Cm', val=np.zeros(nn), desc='Moment coefficient', units=None)

    def setup_partials(self):
        airplane = self.options['airplane_data']
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zz = np.zeros(nn)

        self.declare_partials(of='Cm', wrt='alpha', rows=ar, cols=zz, val=airplane.coeffs.Cmalpha)
        self.declare_partials(of='Cm', wrt='de', rows=ar, cols=ar, val=airplane.coeffs.Cmde)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']
        alpha = inputs['alpha']
        de = inputs['de']

        outputs['Cm'] = airplane.coeffs.Cm0 + airplane.coeffs.Cmalpha * alpha + airplane.coeffs.Cmde * de


class MomentCoeffComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=Airplanes, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']
        zeros = np.zeros(nn)

        self.add_input(name='alpha', val=zeros, desc='Angle of attack', units='rad')
        self.add_input(name='de', val=zeros, desc='Elevator angle', units='rad')
        self.add_input(name='tas', val=zeros, desc='True Airspeed', units='m/s')
        self.add_input(name='q', val=zeros, desc='Pitch Rate', units='rad/s')

        self.add_output(name='Cm', val=zeros, desc='Moment coefficient', units=None)

    def setup_partials(self):
        airplane = self.options['airplane_data']
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.declare_partials(of='Cm', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='Cm', wrt='de', rows=ar, cols=ar)
        self.declare_partials(of='Cm', wrt='tas', rows=ar, cols=ar)
        self.declare_partials(of='Cm', wrt='q', rows=ar, cols=ar)

    def compute(self, inputs, outputs, **kwargs):
        airplane = self.options['airplane_data']
        alpha = inputs['alpha']
        de = inputs['de']
        q = inputs['q']
        tas = inputs['tas']

        qhat = q * airplane.wing.mac / (2 * tas)

        outputs['Cm'] = (
                airplane.coeffs.Cm0
                + airplane.coeffs.Cmalpha * alpha
                + airplane.coeffs.Cmde * de
                + airplane.coeffs.Cmq * qhat
        )

    def compute_partials(self, inputs, partials, **kwargs):
        airplane = self.options['airplane_data']
        q = inputs['q']
        tas = inputs['tas']

        partials['Cm', 'alpha'] = airplane.coeffs.Cmalpha
        partials['Cm', 'de'] = airplane.coeffs.Cmde
        partials['Cm', 'q'] = airplane.coeffs.Cmq * airplane.wing.mac / (2 * tas)
        partials['Cm', 'tas'] = - airplane.coeffs.Cmq * q * airplane.wing.mac / (2 * tas ** 2)
