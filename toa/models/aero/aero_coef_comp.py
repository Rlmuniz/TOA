import numpy as np
import openmdao.api as om
from toa.airplanes import AirplaneData


class AeroCoeffComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('airplane_data', types=AirplaneData, desc='Class containing all airplane data')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='alpha', val=np.zeros(nn), desc='Angle of attack', units='rad')
        self.add_input(name='de', val=np.zeros(nn), desc='Elevator angle', units='rad')
        self.add_input(name='tas', val=np.zeros(nn), desc='True Airspeed', units='m/s')
        self.add_input(name='q', val=np.zeros(nn), desc='Pitch Rate', units='rad/s')

        self.add_output(name='CL', shape=(nn,), desc='Lift coefficient', units=None)
        self.add_output(name='CD', shape=(nn,), desc='Drag coefficient', units=None)
        self.add_output(name='Cm', shape=(nn,), desc='Moment coefficient', units=None)

        # partials
        ar = np.arange(nn)
        self.declare_partials(of='CL', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='de', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='tas', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='q', rows=ar, cols=ar)

        self.declare_partials(of='CD', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CD', wrt='de', rows=ar, cols=ar)
        self.declare_partials(of='CD', wrt='tas', rows=ar, cols=ar)
        self.declare_partials(of='CD', wrt='q', rows=ar, cols=ar)

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

        qhat = q * airplane.cbar / (2 * tas)

        CL = airplane.CL0 + airplane.CLa * alpha + airplane.CLde * de + airplane.CLq * qhat
        outputs['CL'] = CL
        outputs['CD'] = airplane.CDmin + airplane.kCDi * CL ** 2
        outputs['Cm'] = airplane.Cm0 + airplane.Cma * alpha + airplane.Cmde * de + airplane.Cmq * qhat

    def compute_partials(self, inputs, partials, **kwargs):
        airplane = self.options['airplane_data']
        alpha = inputs['alpha']
        de = inputs['de']
        tas = inputs['tas']
        q = inputs['q']

        multi = 2 * airplane.kCDi * (airplane.CL0 + airplane.CLa * alpha + airplane.CLde * de)

        partials['CL', 'alpha'] = airplane.CLa
        partials['CL', 'de'] = airplane.CLde
        partials['CL', 'tas'] = -airplane.CLq * airplane.cbar * q / (2 * tas ** 2)
        partials['CL', 'q'] = airplane.CLq * airplane.cbar / (2 * tas)

        partials['CD', 'alpha'] = multi * partials['CL', 'alpha']
        partials['CD', 'de'] = multi * partials['CL', 'de']
        partials['CD', 'tas'] = multi * partials['CL', 'tas']
        partials['CD', 'q'] = multi * partials['CL', 'q']

        partials['Cm', 'alpha'] = airplane.Cma
        partials['Cm', 'de'] = airplane.Cmde
        partials['Cm', 'tas'] = -airplane.Cmq * airplane.cbar * q / (2 * tas ** 2)
        partials['Cm', 'q'] = airplane.Cmq * airplane.cbar / (2 * tas)
